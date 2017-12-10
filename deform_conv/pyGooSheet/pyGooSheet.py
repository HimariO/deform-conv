
from __future__ import print_function
import httplib2
import os
import getopt
import sys
import datetime
import re
import argparse

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage

parser = argparse.ArgumentParser(parents=[tools.argparser])
parser.add_argument("-sh", "--sheet", help="shpread sheet tab name")
parser.add_argument("-i", "--id", help="shpread sheet id")
parser.add_argument("-s", "--step", help="training step")
parser.add_argument("-l", "--loss", help="loss value(training)")
parser.add_argument("-a", "--accuracy", help="accuracy value(validate)")

args = parser.parse_args()

try:
    import argparse
    # flags = argparse.ArgumentParser()
    flags = args
except ImportError:
    flags = None
# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Google Sheets API Python Quickstart'


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
                                   'sheets.googleapis.com-python-quickstart.json')

    store = Storage(credential_path)
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


def appendValue(service, sheetid, rangeN, values):

    body = {
        'values': values
    }
    result = service.spreadsheets().values().append(
        spreadsheetId=sheetid, range=rangeN,
        valueInputOption='RAW', body=body).execute()
    return result


def clearSheet(service, sheetid, rangeN):
    range_ = rangeN

    clear_values_request_body = {
        # TODO: Add desired entries to the request body.
    }

    request = service.spreadsheets().values().clear(spreadsheetId=sheetid, range=range_, body=clear_values_request_body)
    response = request.execute()
    return response


def createSheet(service, sheetid):

    copy_sheet_to_another_spreadsheet_request_body = {
        'destination_spreadsheet_id': sheetid,
    }

    request = service.spreadsheets().sheets().copyTo(spreadsheetId=sheetid, sheetId=0, body=copy_sheet_to_another_spreadsheet_request_body)
    response = request.execute()
    return response


if __name__ == '__main__':

    """
    If need to resume training from checkpoint, you must start form iteration which have same step with one of npy sample's start file.
    due to teh fact that sample in the middle of file have unknow ID(start + miss_data_num), after program restarting with 0 miss_data_num.
    """
    step = 0
    loss_value = 1

    """Shows basic usage of the Sheets API.

    Creates a Sheets API service object and prints the names and majors of
    students in a sample spreadsheet:
    https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?version=v4')
    service = discovery.build('sheets', 'v4', http=http, discoveryServiceUrl=discoveryUrl)

    spreadsheetId = args.id
    rangeName = args.sheet
    # rangeName = 'Class Data!A2:E'  # 'sheet_name!colume_row_start:colume_row_end'

    values = [
        ['step-%s' % args.step, args.loss, args.accuracy, str(datetime.datetime.now())],
    ]

    update_res = appendValue(service, spreadsheetId, rangeName, values)

    pattern = re.compile(r":\w(\d+)")
    last_row = pattern.search(update_res['updates']['updatedRange']).group(1)
    # print(pattern.search(update_res['updates']['updatedRange']))
    # print('>>>>>>>>>>>>>>> ', last_row)
    # print('>>>>>>>>>>>>>>> ', update_res['updates']['updatedRange'])
    if int(last_row) >= 2000:
        createSheet(service, spreadsheetId)
        clearSheet(service, spreadsheetId, '%s!A2:C3000' % rangeName)
