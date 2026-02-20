from openai import OpenAI, AuthenticationError

# create OpenAI client
def create_client(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()#sanity check listing all availabel model
        return client
    except AuthenticationError:
        print("Incorrect API")
    return None