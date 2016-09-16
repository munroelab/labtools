from __future__ import print_function
import httplib2
import os

from apiclient import discovery
from apiclient import errors
from apiclient.http import MediaFileUpload
import oauth2client
from oauth2client import client
from oauth2client import tools

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/labtools.json
SCOPES = 'https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/drive.appdata'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Lab Tools'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'labtools.json')

    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatibility with Python 2.6
            credentials = tools.run(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def main():
    """Shows basic usage of the Google Drive API.

    Creates a Google Drive API service object and outputs the names and IDs
    for up to 10 files.
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('drive', 'v3', http=http)

    #upload(service)
    print_files_in_folder(service, 'root')
    #list_files(service)

def list_files(service):
    files = service.files()
    request = files.list(pageSize=10,
            #q="name = 'trav1.dat'",
                         spaces='appDataFolder',
                         # fields="nextPageToken, files(id, name, parents)"
                         )

    while request is not None:
        response = request.execute()

        items = response.get('files', [])
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print (item)
                #print('{0} ({1})'.format(item['name'], item['id']))
        
        request = files.list_next(request, response)


def print_files_in_folder(service, folder_id):
  """Print files belonging to a folder.

  Args:
    service: Drive API service instance.
    folder_id: ID of the folder to print files from.
  """
  page_token = None
  while True:
    try:
      param = {}
      if page_token:
        param['pageToken'] = page_token
      children = service.files().list(
          q='root in parents', **param).execute()

      for child in children.get('items', []):
        print('File Id: %s' % child['id'])
      page_token = children.get('nextPageToken')
      if not page_token:
        break
    except errors.HttpError as error:
      print('An error occurred: %s' % error)
      break

def upload(drive_service):
    file_metadata = {
        'name' : 'config.json',
        'parents': [ 'appDataFolder']
    }
    media = MediaFileUpload('files/config.json',
                            mimetype='application/json',
                            resumable=True)
    file = drive_service.files().create(body=file_metadata,
                                        media_body=media,
                                        fields='id').execute()
    print('File ID: %s' % file.get('id'))

if __name__ == '__main__':
    main()
