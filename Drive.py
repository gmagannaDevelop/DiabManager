'''
Drive.py is a single-class module. the Drive class
allows simple manipulation of files. Designed to work
with a single file locally which will be constantly
backed up on Google Drive.
'''
import os

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class Drive(object):
    ''' A simple, single-purpose-specific, wrapper
    for PyDrive.
    '''

    def __init__(self, credentials_file: str = 'mycreds.txt'):
        ''' Initialize the drive object with a default credentials_file,
        which should be in the same directory as the script. A file can be
        specified providing the relative or absolute path.
        'client_secrets.json' MUST BE ON THE SAME DIRECTORY, OTHERWISE
        AN EXCEPTION WILL BE THROWN.
        '''
        if 'client_secrets.json' not in os.listdir('.'):
            raise Exception
        self.__gauth = GoogleAuth()
        try:
            self.__gauth.LoadCredentialsFile(credentials_file)
        except Exception:
            pass
        if self.__gauth.credentials is None:
            # Platform-specific handling of missing credentials.
            if os.uname().sysname == 'Linux':
                self.__gauth.LocalWebserverAuth()
            elif (os.uname().sysname == 'Darwin' and\
                    'iPhone' in os.uname().machine):
                import console
                console.alert('ERROR: Manual authentication needed.')
                self.__gauth.LocalWebserverAuth()
            else:
                raise Exception
        elif self.__gauth.access_token_expired:
            self.__gauth.Refresh()
        else:
            self.__gauth.Authorize()
        self.__gauth.SaveCredentialsFile(credentials_file)
        self.__drive = GoogleDrive(self.__gauth)
    # END __init__

    @property
    def drive(self):
        return self.__drive

    def get_file_id(self, file_name: str = ''):
        ''' Get the file id of the desired file, if it exists.
        Return False upon failure i.e. file not specified, file doesn't
        exist, etc.
        '''
        if not file_name:
            return False
        file_list = self.__query_drive()
        names = [_file['title'] for _file in file_list]
        ids = [_file['id'] for _file in file_list]
        if file_name in names:
            return ids[names.index(file_name)]
        else:
            return False

    def get_file_by_name(self, file_name: str = ''):
        ''' Get a GoogleDriveFile instance corresponding to the
        specified file_name, if it exists.
        Return False upon failure i.e. file doesn't exist, etc.
        '''
        if not file_name:
            raise Exception('file_name parameter missing.')
        file_list = self.__query_drive()
        names = [_file['title'] for _file in file_list]
        if file_name in names:
            return file_list[names.index(file_name)]
        else:
            return False

    def download(self, file_name: str = '', target_name: str = ''):
        ''' Download file from drive.
        Query GoogleDrive for file_name.
        Save file to targe_name, if specified.
        target_name defaults to file_name (used to query).

        Returns:
            True, upon success
            False, upon failure
        '''
        if not file_name:
            raise Exception('file_name parameter missing.')
        else:
            _file = self.get_file_by_name(file_name)
        if not target_name:
            target_name = file_name

        if _file:
            _file.GetContentFile(target_name)
            return True
        else:
            return False

    def file_exists(self, some_file: str, query: str = '') -> bool:
        ''' Query Drive to verify the existence of a given file.
        The provided string 'some_file' should correpond EXACTLY
        to the name that appears on GoogleDrive.
        If no query is provided, the default _query will yield
        all files in the root folder.
        Useful links on query syntax:
            https://pythonhosted.org/PyDrive/filelist.html
            https://developers.google.com/drive/api/v2/search-parameters
        '''
        file_list = self.__query_drive(query)
        if some_file in [_file['title'] for _file in file_list]:
            return True
        else:
            return False

    def update(self, file_name: str, path: str = '') -> bool:
        ''' Update a file stored on Google Drive, using a
        local file. If the file does not exist on Google Drive,
        a new Google Drive file is created and its content is
        set to the specified local file's content.
        This method UPLOADS the file, with all of its content.
        Appending one line to a 7GiB file will result in the
        uploading of 7GiB + sizeof(one line).

        Returns:
                True    (successful uploading)
                False   (if any error occurs)
        '''
        file_list = self.__query_drive()
        titles = [_file['title'] for _file in file_list]
        if path:
            path_to_file = os.path.join(path, file_name)
        else:
            path_to_file = file_name
        if file_name in titles:
            _index = titles.index(file_name)
            _gdrive_file = file_list[_index]
        else:
            _gdrive_file = self.__drive.CreateFile({'title': file_name})
        try:
            _gdrive_file.SetContentFile(path_to_file)
            _gdrive_file.Upload()
            return True
        except (BaseException, FileNotFoundError):
            return False

    def __query_drive(self, query: str = '') -> list:
        ''' Helper method returning a list of files.
        A wrapper for the call:
            self.__drive.ListFile(_query).GetList()
        Default query:
            {'q': "'root' in parents and trashed=false"}
        '''
        if query:
            _query = query
        else:
            _query = {'q': "'root' in parents and trashed=false"}
        file_list = self.__drive.ListFile(_query).GetList()
        return file_list

# END Drive
