"""
blur_utils.py
=============

Function library for local face detection and face blurring (images are never
sent to any external service).

Two face detectors are provided:

- CenterFace (ONNX model, run through OpenCV's ``cv2.dnn``)
- YuNet (ONNX model, run through OpenCV's ``cv2.FaceDetectorYN``)

The blurring itself is shared between both detectors: a Gaussian blur (with an
optional elliptical mask) is applied to every face bounding box returned by the
detector.

Dependencies: ``opencv-python``, ``numpy`` (on top of Pillow, already used in
the project).

Usage example
-------------
>>> from src.blur_utils import blur_folder
>>> blur_folder("data/image_positive_controls", "results/blurred_images/centerface",
...             method="centerface")
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np


# =====================================================================
# Models: download URLs and on-demand caching
# =====================================================================

# Default location where the ONNX models are cached (git-ignored).
MODELS_DIR = Path(__file__).resolve().parent.parent / "data_utils" / "models"

# Public sources of the pre-trained models.
CENTERFACE_URL = (
    "https://github.com/Star-Clouds/CenterFace/raw/master/models/onnx/centerface.onnx"
)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def download_model(url: str, dest_path: Path) -> Path:
    """Download an ONNX model unless it is already present locally.

    Parameters
    ----------
    url : str
        URL of the ``.onnx`` file to download.
    dest_path : Path
        Local destination path.

    Returns
    -------
    Path
        Local path of the model (downloaded or already existing).
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        print(f"Downloading model to {dest_path} ...")
        urllib.request.urlretrieve(url, dest_path)
        print("Done.")
    return dest_path


def get_centerface_model(models_dir: Path = MODELS_DIR) -> Path:
    """Return the CenterFace model path, downloading it if needed."""
    return download_model(CENTERFACE_URL, Path(models_dir) / "centerface.onnx")


def get_yunet_model(models_dir: Path = MODELS_DIR) -> Path:
    """Return the YuNet model path, downloading it if needed."""
    return download_model(YUNET_URL, Path(models_dir) / "face_detection_yunet_2023mar.onnx")


# =====================================================================
# Image path handling (inspired by preparatory_report3.ipynb)
# =====================================================================

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def get_image_paths(image_folder, recursive: bool = False):
    """Return the sorted image paths contained in a folder.

    Reuses the loading logic from ``preparatory_report3.ipynb`` (jpg / jpeg /
    png), with an extra recursive option.

    Parameters
    ----------
    image_folder : str | Path
        Folder containing the images.
    recursive : bool
        If True, also walks sub-folders.

    Returns
    -------
    list[Path]
        Sorted list of image paths.
    """
    folder = Path(image_folder)
    paths = []
    # glob is case-sensitive, and the dataset mixes .png/.PNG/.jpg/.JPG, so we
    # match each extension in both cases and de-duplicate afterwards.
    for ext in IMAGE_EXTENSIONS:
        globber = folder.rglob if recursive else folder.glob
        paths += list(globber(ext)) + list(globber(ext.upper()))
    # Drop duplicates (case-insensitive filesystems may match twice) and sort.
    return sorted(set(paths))


def filter_paths_by_stem(paths, stems):
    """Keep only the paths whose file stem (name without extension) is selected.

    Selecting by stem makes the selection robust to extension differences
    (e.g. ``36.jpg`` vs ``36.PNG``): a requested number matches every file
    sharing that stem.

    Parameters
    ----------
    paths : list[Path]
        Candidate image paths.
    stems : iterable
        Stems to keep, as strings (e.g. ``["6", "8", "36"]``).

    Returns
    -------
    list[Path]
        Filtered, sorted list of paths.
    """
    wanted = {str(s) for s in stems}
    return sorted(p for p in paths if p.stem in wanted)


# =====================================================================
# Detector 1: CenterFace (ONNX via OpenCV cv2.dnn)
# =====================================================================

class CenterFace:
    """CenterFace face detector running from an ONNX model.

    Standard implementation (Star-Clouds/CenterFace): the network returns a
    heatmap, a scale map, an offset map and landmarks, which are then decoded
    into bounding boxes.
    """

    def __init__(self, model_path=None, landmarks: bool = True):
        if model_path is None:
            model_path = get_centerface_model()
        self.landmarks = landmarks
        # Run the model through OpenCV's DNN module rather than onnxruntime:
        # the published centerface.onnx declares a fixed input shape, which
        # onnxruntime enforces, whereas OpenCV reshapes the network to the
        # actual blob size on every call (so arbitrary image sizes work).
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.img_h_new = self.img_w_new = 0
        self.scale_h = self.scale_w = 1.0

    def __call__(self, img, threshold: float = 0.5):
        """Detect faces on a BGR image (numpy array).

        Returns
        -------
        (dets, lms) if ``landmarks`` else ``dets``.
        ``dets`` is an (N, 5) array: x1, y1, x2, y2, score.
        """
        height, width = img.shape[:2]
        self.img_h_new, self.img_w_new, self.scale_h, self.scale_w = self._transform(height, width)
        return self._inference(img, threshold)

    @staticmethod
    def _transform(h, w):
        # CenterFace requires input dimensions that are multiples of 32.
        img_h_new, img_w_new = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        scale_h, scale_w = img_h_new / h, img_w_new / w
        return img_h_new, img_w_new, scale_h, scale_w

    def _inference(self, img, threshold):
        # Build the network input blob (resized to a multiple of 32, BGR->RGB).
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0,
            size=(self.img_w_new, self.img_h_new),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        # Identify outputs by channel count rather than by name (names differ
        # between model exports): heatmap=1, scale=2, offset=2, landmarks=10.
        heatmap, scale, offset, lms = self._sort_outputs(outputs)
        dets, lms = self._decode(
            heatmap, scale, offset, lms, (self.img_h_new, self.img_w_new), threshold
        )
        # Rescale boxes (and landmarks) back to the original image size.
        if len(dets) > 0:
            dets[:, 0:4:2] = dets[:, 0:4:2] / self.scale_w
            dets[:, 1:4:2] = dets[:, 1:4:2] / self.scale_h
            if self.landmarks and len(lms) > 0:
                lms[:, 0:10:2] = lms[:, 0:10:2] / self.scale_w
                lms[:, 1:10:2] = lms[:, 1:10:2] / self.scale_h
        else:
            dets = np.empty(shape=[0, 5], dtype=np.float32)
            lms = np.empty(shape=[0, 10], dtype=np.float32)
        return (dets, lms) if self.landmarks else dets

    @staticmethod
    def _sort_outputs(outputs):
        # Map the four network outputs to (heatmap, scale, offset, landmarks)
        # using their channel count. The two 2-channel maps (scale, offset)
        # keep their network order, which is scale first then offset.
        heatmap = lms = None
        two_channel = []
        for o in outputs:
            channels = o.shape[1]
            if channels == 1:
                heatmap = o
            elif channels == 10:
                lms = o
            elif channels == 2:
                two_channel.append(o)
        scale, offset = two_channel[0], two_channel[1]
        return heatmap, scale, offset, lms

    def _decode(self, heatmap, scale, offset, landmark, size, threshold):
        # Decode the network outputs into bounding boxes above the threshold.
        heatmap = np.squeeze(heatmap)
        scale0, scale1 = scale[0, 0, :, :], scale[0, 1, :, :]
        offset0, offset1 = offset[0, 0, :, :], offset[0, 1, :, :]
        c0, c1 = np.where(heatmap > threshold)
        boxes, lms = [], []
        if len(c0) > 0:
            for i in range(len(c0)):
                # Decoded box size from the (log) scale maps; stride is 4.
                s0 = np.exp(scale0[c0[i], c1[i]]) * 4
                s1 = np.exp(scale1[c0[i], c1[i]]) * 4
                o0, o1 = offset0[c0[i], c1[i]], offset1[c0[i], c1[i]]
                s = heatmap[c0[i], c1[i]]
                x1 = max(0, (c1[i] + o1 + 0.5) * 4 - s1 / 2)
                y1 = max(0, (c0[i] + o0 + 0.5) * 4 - s0 / 2)
                x1, y1 = min(x1, size[1]), min(y1, size[0])
                boxes.append([x1, y1, min(x1 + s1, size[1]), min(y1 + s0, size[0]), s])
                if self.landmarks:
                    lm = []
                    for j in range(5):
                        lm.append(landmark[0, j * 2 + 1, c0[i], c1[i]] * s1 + x1)
                        lm.append(landmark[0, j * 2, c0[i], c1[i]] * s0 + y1)
                    lms.append(lm)
            boxes = np.asarray(boxes, dtype=np.float32)
            # Non-maximum suppression to remove overlapping detections.
            keep = self._nms(boxes[:, :4], boxes[:, 4], 0.3)
            boxes = boxes[keep, :]
            if self.landmarks:
                lms = np.asarray(lms, dtype=np.float32)[keep, :]
        return boxes, lms

    @staticmethod
    def _nms(boxes, scores, nms_thresh):
        # Greedy non-maximum suppression (IoU based).
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)[::-1]
        suppressed = np.zeros(boxes.shape[0], dtype=bool)
        keep = []
        for _i in range(boxes.shape[0]):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(i)
            for _j in range(_i + 1, boxes.shape[0]):
                j = order[_j]
                if suppressed[j]:
                    continue
                xx1, yy1 = max(x1[i], x1[j]), max(y1[i], y1[j])
                xx2, yy2 = min(x2[i], x2[j]), min(y2[i], y2[j])
                w, h = max(0, xx2 - xx1 + 1), max(0, yy2 - yy1 + 1)
                inter = w * h
                if inter / (areas[i] + areas[j] - inter) >= nms_thresh:
                    suppressed[j] = True
        return keep


# =====================================================================
# Blurring: helper shared by both detectors
# =====================================================================

def _apply_blur(img, boxes, mask_scale: float = 1.3, blur_factor: float = 3.0,
                ellipse: bool = True):
    """Apply a Gaussian blur to every face bounding box.

    Parameters
    ----------
    img : np.ndarray
        BGR image (modified in place and returned).
    boxes : iterable
        Boxes ``[x1, y1, x2, y2, ...]`` (extra columns are ignored).
    mask_scale : float
        Box enlargement around the face (1.0 = exact box).
    blur_factor : float
        Blur strength; the Gaussian sigma is ``blur_factor * face_width / 100``.
    ellipse : bool
        If True, apply an elliptical mask (more natural blur); otherwise blur
        the whole rectangle.

    Returns
    -------
    np.ndarray
        The blurred image.
    """
    h, w = img.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        # Enlarge the box around the face center.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * mask_scale, (y2 - y1) * mask_scale
        x1, x2 = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
        y1, y2 = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = img[y1:y2, x1:x2]
        # Sigma proportional to the face size; kernel size must be odd.
        sigma = max(blur_factor * (x2 - x1) / 100.0, 1.0)
        ksize = int(sigma * 4) | 1  # force odd
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma)

        if ellipse:
            # Blend the blurred region through an elliptical mask.
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            center = (roi.shape[1] // 2, roi.shape[0] // 2)
            axes = (roi.shape[1] // 2, roi.shape[0] // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask3 = mask[:, :, None].astype(bool)
            roi[:] = np.where(mask3, blurred, roi)
        else:
            roi[:] = blurred
    return img


# =====================================================================
# Function 1: blurring with CenterFace
# =====================================================================

# Detector cache to avoid reloading the model for every image.
_centerface_detector = None


def blur_faces_centerface(image_path, model_path=None, threshold: float = 0.5,
                          mask_scale: float = 1.3, blur_factor: float = 3.0,
                          ellipse: bool = True):
    """Detect and blur the faces of an image with **CenterFace** (ONNX).

    Parameters
    ----------
    image_path : str | Path
        Path of the image to process.
    model_path : str | Path | None
        Path of the ``centerface.onnx`` model. If None, it is downloaded/cached
        automatically.
    threshold : float
        Detection confidence threshold.
    mask_scale, blur_factor, ellipse :
        Blurring parameters (see ``_apply_blur``).

    Returns
    -------
    (np.ndarray, np.ndarray)
        The blurred BGR image and the detected boxes (N, 5).
    """
    global _centerface_detector
    if _centerface_detector is None:
        _centerface_detector = CenterFace(model_path=model_path, landmarks=False)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    dets = _centerface_detector(img, threshold=threshold)
    img = _apply_blur(img, dets, mask_scale=mask_scale, blur_factor=blur_factor, ellipse=ellipse)
    return img, dets


# =====================================================================
# Function 2: blurring with YuNet
# =====================================================================

def blur_faces_yunet(image_path, model_path=None, score_threshold: float = 0.6,
                     nms_threshold: float = 0.3, mask_scale: float = 1.3,
                     blur_factor: float = 3.0, ellipse: bool = True):
    """Detect and blur the faces of an image with **YuNet** (OpenCV).

    Parameters
    ----------
    image_path : str | Path
        Path of the image to process.
    model_path : str | Path | None
        Path of the YuNet ``.onnx`` model. If None, it is downloaded/cached
        automatically.
    score_threshold, nms_threshold : float
        Detection and non-maximum-suppression thresholds.
    mask_scale, blur_factor, ellipse :
        Blurring parameters (see ``_apply_blur``).

    Returns
    -------
    (np.ndarray, np.ndarray)
        The blurred BGR image and the detected boxes (N, 4): x1, y1, x2, y2.
    """
    if model_path is None:
        model_path = get_yunet_model()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]

    # YuNet needs to know the input size before detection.
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(w, h),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)

    boxes = []
    if faces is not None:
        # YuNet returns (x, y, w, h, ...landmarks..., score) -> convert to x1,y1,x2,y2.
        for f in faces:
            x, y, fw, fh = f[:4]
            boxes.append([x, y, x + fw, y + fh])
    boxes = np.asarray(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)

    img = _apply_blur(img, boxes, mask_scale=mask_scale, blur_factor=blur_factor, ellipse=ellipse)
    return img, boxes


# =====================================================================
# Batch processing: blur a list of images / a folder and save the results
# =====================================================================

def blur_image_paths(image_paths, output_folder, method: str = "centerface", **kwargs):
    """Blur an explicit list of images and save them, keeping their file names.

    Parameters
    ----------
    image_paths : iterable[str | Path]
        Images to process.
    output_folder : str | Path
        Destination folder (created if needed).
    method : {"centerface", "yunet"}
        Detector to use.
    **kwargs :
        Parameters forwarded to the chosen blurring function.

    Returns
    -------
    list[dict]
        One summary per image: ``{"file": ..., "n_faces": ...}``.
    """
    blur_fn = {"centerface": blur_faces_centerface, "yunet": blur_faces_yunet}[method]

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    summary = []
    for p in image_paths:
        p = Path(p)
        img, boxes = blur_fn(p, **kwargs)
        out_path = output_folder / p.name
        cv2.imwrite(str(out_path), img)
        summary.append({"file": p.name, "n_faces": int(len(boxes))})
        print(f"[{method}] {p.name}: {len(boxes)} face(s) -> {out_path}")
    return summary


def blur_folder(input_folder, output_folder, method: str = "centerface",
                recursive: bool = False, limit: int = None, **kwargs):
    """Blur all the images of a folder and save them.

    Thin wrapper around :func:`blur_image_paths` that first lists the folder.

    Parameters
    ----------
    input_folder : str | Path
        Source folder.
    output_folder : str | Path
        Destination folder (created if needed). File names are kept.
    method : {"centerface", "yunet"}
        Detector to use.
    recursive : bool
        Also walk sub-folders.
    limit : int | None
        If set, only process the first ``limit`` images (sub-sample).
    **kwargs :
        Parameters forwarded to the chosen blurring function.

    Returns
    -------
    list[dict]
        One summary per image: ``{"file": ..., "n_faces": ...}``.
    """
    paths = get_image_paths(input_folder, recursive=recursive)
    if limit is not None:
        paths = paths[:limit]
    return blur_image_paths(paths, output_folder, method=method, **kwargs)
