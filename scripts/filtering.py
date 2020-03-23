
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation

from utils.color import getRandomColor
from scripts.embedding import aggregateApiSequences, trainW2Vmodel
from scripts.preprocessing import mappingApiNormalize, \
    apiStat, removeApiRedundance, filterApiSequence, statApiFrequency
from utils.file import loadJson, dumpJson
from utils.display import printState

base_path = 'D:/peimages/PEs/virushare_20/'
path = base_path + 'jsons/'
apiSetPath = base_path + 'api_set.json'
reportPath = base_path + 'json_w_e_report.json'
original_word2cluster_path = base_path + 'ApiClusterTable(ori).json'
real_word2cluster_path = base_path + 'ApiClusterTable(subs).json'


#-------------------------统计---------------------------------
# statApiFrequency(json_path=path,
#                  is_class_dir=False,
#                  threshold=5e-3)
#-------------------------------------------------------------


#-------------------------过滤---------------------------------
# filterApiSequence(json_path=path,
#                   filterd_apis=['NtOpenKeyEx', 'NtOpenDirectoryObject', 'SHGetFolderPathW', 'NtOpenProcess', 'FindWindowA', 'OutputDebugStringA', 'LoadResource', 'FindWindowExA', 'InternetReadFile', 'CopyFileExW', 'LoadStringA', 'LdrGetDllHandle', 'Thread32Next', 'NtProtectVirtualMemory', 'NtOpenKey', 'LdrUnloadDll', 'WriteConsoleW', 'NtCreateMutant', 'GetCursorPos', 'CryptDecrypt', 'NtSetInformationFile', 'FindResourceA', 'GetFileSizeEx', 'NtQueryInformationFile', 'NtMapViewOfSection', 'RegEnumKeyW', 'RegQueryValueExA', 'EnumWindows', 'DrawTextExW', 'CoInitializeEx', 'CoUninitialize', 'NtUnmapViewOfSection', 'CoCreateInstance', 'GetFileAttributesExW', 'RegSetValueExA', 'LoadStringW', 'GetVolumeNameForVolumeMountPointW', 'NtQuerySystemInformation', 'NtCreateSection', 'RegCreateKeyExA', 'SizeofResource', 'SetFilePointerEx', 'NtOpenMutant', 'RegEnumKeyExW', 'NtTerminateProcess', 'NtQueryAttributesFile', 'CreateToolhelp32Snapshot', 'FindWindowW', 'FindResourceExA', 'SetEndOfFile', 'RegEnumValueA', 'GetSystemInfo', 'NtDuplicateObject', 'RegEnumValueW', 'GetShortPathNameW', 'Process32FirstW', 'NtReadVirtualMemory', 'RegEnumKeyExA', 'GetSystemDirectoryA', 'SearchPathW', 'GetUserNameA', 'recvfrom', 'Module32NextW', 'FindResourceW', 'GetSystemWindowsDirectoryW', 'CreateDirectoryW', 'NtOpenSection', 'RegCreateKeyExW', 'NtEnumerateValueKey', 'SetFileAttributesW', 'CreateActCtxW', 'select', 'GetVolumePathNamesForVolumeNameW', 'GetTempPathW', 'StartServiceA', 'CreateProcessInternalW', 'CoGetClassObject', 'NtEnumerateKey', 'GetKeyboardState', 'RegDeleteValueA', 'SetFileTime', 'RegQueryInfoKeyW', 'CopyFileA', 'RegSetValueExW', 'CreateThread', 'IsDebuggerPresent', 'OleInitialize', 'NtResumeThread', 'GetFileInformationByHandleEx', 'GlobalMemoryStatus', 'GetSystemDirectoryW', 'OpenServiceA', 'OpenSCManagerA', 'GetTimeZoneInformation', 'LookupPrivilegeValueW', 'InternetSetOptionA', 'RegQueryInfoKeyA', 'SetWindowsHookExA', 'UnhookWindowsHookEx', 'NtSetValueKey', 'SetStdHandle', 'CryptAcquireContextW', 'NtCreateKey', 'CryptDecodeObjectEx', 'NtOpenThread', 'CreateDirectoryExW', 'SetUnhandledExceptionFilter', 'DrawTextExA', 'GetComputerNameA', 'RegDeleteValueW', 'FindWindowExW', 'GetDiskFreeSpaceExW', 'NtSuspendThread', 'DeviceIoControl', 'closesocket', 'UuidCreate', 'InternetCloseHandle', 'NtGetContextThread', 'CopyFileW', 'getaddrinfo', 'WSAStartup', 'socket', 'GetNativeSystemInfo', 'GetFileVersionInfoSizeW', 'ShellExecuteExW', 'GetFileVersionInfoW', 'SendNotifyMessageW', 'gethostbyname', 'SetFileInformationByHandle', 'ioctlsocket', 'GetAdaptersAddresses', 'InternetGetConnectedState', 'InternetQueryOptionA', 'InternetCrackUrlA', 'GlobalMemoryStatusEx', 'LookupAccountSidW', 'ControlService', 'connect', 'GetComputerNameW', 'GetFileInformationByHandle', 'Module32FirstW', 'setsockopt', 'InternetOpenA', 'CWindow_AddTimeoutCode', 'SHGetSpecialFolderLocation', 'WriteConsoleA', 'NtCreateThreadEx', 'GetFileVersionInfoSizeExW', 'GetFileVersionInfoExW', 'CoCreateInstanceEx', 'GetVolumePathNameW', 'GetUserNameW', 'CreateRemoteThread', 'OpenSCManagerW', 'OpenServiceW', 'InternetOpenW', 'SetWindowsHookExW', 'NtWriteVirtualMemory', 'MessageBoxTimeoutA', 'InternetCrackUrlW', 'InternetConnectW', 'HttpSendRequestW', 'HttpOpenRequestW', 'GetUserNameExA', 'CoInitializeSecurity', 'shutdown', 'HttpQueryInfoA', 'RegDeleteKeyA', 'RtlAddVectoredExceptionHandler', 'NtQueryFullAttributesFile', 'NtSetContextThread', 'CryptExportKey', 'WSASocketW', 'bind', 'GetBestInterfaceEx', 'RtlRemoveVectoredExceptionHandler', 'NtTerminateThread', 'GetDiskFreeSpaceW', 'CryptAcquireContextA', 'InternetOpenUrlA', 'ReadCabinetState', 'InternetConnectA', 'NtQueueApcThread', 'GetSystemWindowsDirectoryA', 'Thread32First', 'getsockname', 'RtlDecompressBuffer', 'HttpOpenRequestA', 'IWbemServices_ExecQuery', 'HttpSendRequestA', 'CertOpenStore', 'GetAdaptersInfo', 'InternetOpenUrlW', 'GetAddrInfoW', 'WSASocketA', 'MoveFileWithProgressW', 'GetUserNameExW', 'WSARecv', 'CDocument_write', 'WSASend', 'NtDeleteKey', 'InternetSetStatusCallback', 'sendto', 'URLDownloadToFileW', 'RegDeleteKeyW', 'RtlAddVectoredContinueHandler', 'NetShareEnum', 'RemoveDirectoryA', 'AssignProcessToJobObject', 'CreateServiceA', 'CertControlStore', 'listen', 'accept', 'CreateJobObjectW', 'DnsQuery_A', 'CIFrameElement_CreateElement', 'CryptEncrypt', 'RemoveDirectoryW', 'DeleteService', 'send', 'WSASendTo', 'SetInformationJobObject', 'ObtainUserAgentString', 'WSAConnect', 'StartServiceW', 'CImgElement_put_src', 'WSAAccept', 'NtDeleteValueKey', 'recv', 'COleScript_Compile', 'NetUserGetInfo', 'Math_random', 'CryptUnprotectData', 'CElement_put_innerHTML', 'DeleteUrlCacheEntryA', 'CryptProtectMemory', 'CryptGenKey', 'ActiveXObjectFncObj_Construct', 'CryptUnprotectMemory', 'IWbemServices_ExecMethod', 'CreateServiceW', '__anomaly__', 'NtQueryMultipleValueKey', 'RtlCreateUserThread', 'RtlCompressBuffer', 'NtSaveKey', 'SendNotifyMessageA', 'WSARecvFrom', 'DeleteUrlCacheEntryW', 'CScriptElement_put_src', 'EnumServicesStatusA', 'CryptProtectData', 'CertOpenSystemStoreA', 'NtLoadKey', 'PRF', 'NtDeleteFile', 'InternetGetConnectedStateExW', 'ExitWindowsEx', 'CHyperlink_SetUrlComponent', 'NetUserGetLocalGroups', 'NetGetJoinInformation', 'FindFirstFileExA', 'RegisterHotKey', 'system', 'InternetGetConnectedStateExA', 'CertCreateCertificateContext'])
#-------------------------------------------------------------

#-------------------------查看长度---------------------------------
# apiStat(path=path,
#         dump_apiset_path=apiSetPath,
#         dump_report_path=reportPath,
#         ratio_stairs=[100, 200, 500, 600, 1000, 2000, 3000, 5000, 7000, 10000],
#         class_dir=False)
# -------------------------------------------------------------

#-------------------------API聚类过滤---------------------------------
# 收集API集合
printState('Collecting APIs...')
apiStat(path,
        dump_apiset_path=apiSetPath,
        class_dir=False)

# 制作API名称到名称下标的映射，用于减少内存消耗
api_set = loadJson(apiSetPath)['api_set']
apiMap = {name:str(i) for i,name in enumerate(api_set)}

# 使用映射替换数据集中的API名称
printState('Mapping...')
mappingApiNormalize(path, mapping=apiMap)

printState('Aggregating...')
seqs = aggregateApiSequences(path, is_class_dir=False)

printState('Training Word2Vector')
matrix, word2idx = trainW2Vmodel(seqs,
                                 size=32,
                                 padding=False)

plt.scatter(matrix[:,0], matrix[:,1])
plt.show()

# 聚类
printState('Clustering...')
af = AffinityPropagation().fit(matrix)

#
# word2cluster = {}
# for word, idx in word2idx.items():
#     word2cluster[word] = str(af.labels_[idx] + 1)   # 由于要给<PAD>留一个位置，因此类簇下标要加1
#
# word2cluster_org = {}
# for org_word, subs_word in apiMap.items():
#     word2cluster_org[org_word] = word2cluster[subs_word]
#
# dumpJson(word2cluster, real_word2cluster_path)
# dumpJson(word2cluster_org, original_word2cluster_path)
#
# printState('Mapping word to cluster index...')
# mappingApiNormalize(path,
#                     mapping=word2cluster)
#
# printState('Removing redundancy...')
# removeApiRedundance(path,selected_apis=None)
#
# printState('Done')
#-------------------------------------------------------------






