import zdrive
from zdrive import Downloader
import pandas as pd
import pickle
import os.path
from alive_progress import alive_bar
import time
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# scopes to allow for access to google drive and google sheets

class drive_hookup:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        self.creds = self.gsheet_api_check(self.scopes)

    def gsheet_api_check(self, SCOPES):
        """
        create credentials
        """
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        return creds

    def pull_sheet_data(self, SPREADSHEET_ID,DATA_TO_PULL):
        """
        pull google sheet data
        """

        service = build('sheets', 'v4', credentials=self.creds)
        sheet = service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=DATA_TO_PULL).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
        else:
            rows = sheet.values().get(spreadsheetId=SPREADSHEET_ID,
                                      range=DATA_TO_PULL).execute()
            data = rows.get('values')
            df = pd.DataFrame(data[1:], columns=data[0])
            print("COMPLETE: Data copied")
        return df

    def get_files_in_folder(self, folder_id):
        service = build('drive', 'v3', credentials=self.creds)
        items = []
        topFolderId = folder_id
        pageToken = ""
        while pageToken is not None:
            response = service.files().list(q="'" + topFolderId + "' in parents", pageSize=1000, pageToken=pageToken, fields="nextPageToken, files(id, name)").execute()
            items.extend(response.get('files', []))
            pageToken = response.get('nextPageToken')
        return items

    def find_missing_file(self):
        service = build('drive', 'v3', credentials=self.creds)
        #logs = service.files().list(q="(mimeType='text/x-log' or mimeType='text/plain') and trashed=false",pageSize=1000).execute()
        logs = service.files().list(q="name='10282020-181822_Air_Fly5.log' and trashed=false",pageSize=1000).execute()
        return logs

    def copy_all_log_files(self):
        service = build('drive', 'v3', credentials=self.creds)
        # logs = service.files().list(q="mimeType='text/x-log' and trashed=false",pageSize=1000).execute()

        # get all the log files in the Google Drive
        logs = service.files().list(q="(mimeType='text/x-log' or mimeType='text/plain') and trashed=false",pageSize=1000).execute()
        df_logs = pd.DataFrame(logs.get('files', []))
        df_logs = df_logs.drop(['kind', 'mimeType'], axis=1)

        # get logs in 'All_behavioral_log_files'
        data_folder = '1BI9H_VuAzrL2j9oCZU6PjWt5UYEj5T68'
        folder_files = self.get_files_in_folder(data_folder)
        df_dest = pd.DataFrame(folder_files)

        # dataframe with files that don't already exist in 'All_behavioral_log_files'
        df_copy = df_logs[~df_logs.name.isin(df_dest.name)]

        # make copies
        with alive_bar(len(df_copy)) as bar:
            for row in zip(df_copy['name'], df_copy['id']):
                file_name = row[0]
                file_id = row[1]
                print('copying file: ', file_name, file_id)
                service.files().copy(fileId=file_id, body={"parents": [data_folder], 'name': file_name} ).execute()
                bar()

        return logs

    def download_folder(self, folder_id, destination):
        d = Downloader()
        d.downloadFolder(folder_id, destinationFolder=destination)

    def download_logs_to_local(self, local_folder):
        d = Downloader()
        # first, update the Google Drive folder containing all logs ('All_behavioral_log_files')
        self.copy_all_log_files()

        # files in 'All_behavioral_log_files'
        data_folder = '1BI9H_VuAzrL2j9oCZU6PjWt5UYEj5T68'
        folder_files = self.get_files_in_folder(data_folder)
        df_logs = pd.DataFrame(folder_files)

        # files in local
        local_logs = []
        for file in os.listdir(local_folder):
            # check only log files
            if file.endswith('.log'):
                local_logs.append(file)
        df_local = pd.DataFrame({'name':local_logs})

        # dataframe with files that don't already exist in the local folder
        df_copy = df_logs[~df_logs.name.isin(df_local.name)]

        # make copies
        for row in zip(df_copy['name'], df_copy['id']):
            file_name = os.path.join(local_folder, row[0])
            print(file_name)
            file_id = row[1]
            d.downloadFile(file_id, filePath=file_name)


if __name__ == '__main__':
    d = drive_hookup()
    # d.copy_all_log_files()
    # d.download_logs_to_local(local_folder = '/Volumes/Andy/logs')
