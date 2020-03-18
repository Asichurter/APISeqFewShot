# API序列数据预处理

## 1. 名称统一化
   对于一些名称相似的调用，如CopyFileWx和CopyFileA，使用统一的名称，如CopyFile。映射列表如下：
    
    "RegCreateKeyExA" : "RegCreateKey",
    
    "RegCreateKeyExW" : "RegCreateKey",
    
    "RegDeleteKeyA" : "RegDeleteKey",
    
    "RegDeleteKeyW" : "RegDeleteKey",
    
    "RegSetValueExA" : "RegSetValue",
    
    "RegSetValueExW" : "RegSetValue",
    
    "RegDeleteValueW" : "RegDeleteValue",
    
    "RegDeleteValueA" : "RegDeleteValue",
    
    "RegEnumValueW" : "RegEnumValue",
    
    "RegEnumValueA" : "RegEnumValue",
    
    "RegQueryValueExW" : "RegQueryValue",
    
    "RegQueryValueExA" : "RegQueryValue",
    
    "CreateProcessInternalW" : "CreateProcess",
    
    "NtCreateThreadEx" : "NtCreateThread",
    
    "CreateRemoteThread" : "CreateRemoteThread",
    
    "CreateThread" : "CreateThread",
    
    "NtTerminateProcess": "TerminateProcess",
    
    "NtOpenProcess" : "OpenProcess",
    
    "InternetOpenUrlA" : "InternetOpenUrl",
    
    "InternetOpenUrlW" : "InternetOpenUrl",
    
    "InternetOpenW" : "InternetOpen",
    
    "InternetOpenA" : "InternetOpen",
    
    "InternetConnectW" : "InternetConnect",
    
    "InternetConnectA" : "InternetConnect",
    
    "HttpOpenRequestW" : "HttpOpenRequest",
    
    "HttpOpenRequestA" : "HttpOpenRequest",
    
    "HttpSendRequestA" : "HttpSendRequest",
    
    "HttpSendRequestW" : "HttpSendRequest",
    
    "ShellExecuteExW" : "ShellExecute",
    
    "LdrLoadDll" : "LdrLoadDll",
    
    "CopyFileW" : "CopyFile",
    
    "CopyFileA" : "CopyFile",
    
    "CopyFileExW" : "CopyFile",
    
    "NtCreateFile" : "CreateFile",
    
    "DeleteFileW" : "DeleteFile",
    
    "NtDeleteFile" : "NtDeleteFile",

## 2. 选择特定的监视API
只监视部分重要的API。列表来自论文
"Learning Malware Representation based on Execution Sequences" 和
"Tagging Malware Intentions by Using Attention-Based Sequence-to-Sequence Neural Network"。

监视列表如下：

    "RegCreateKey",
    "RegDeleteKey",
    "RegSetValue",
    "RegDeleteValue",
    "RegEnumValue",
    "RegQueryValue",
    "CreateProcess",
    "NtCreateThread",
    "CreateRemoteThread",
    "CreateThread",
    "TerminateProcess",
    "OpenProcess",
    "InternetOpenUrl",
    "InternetOpen",
    "InternetConnect",
    "HttpOpenRequest",
    "HttpSendRequest",
    "ShellExecute",
    "LdrLoadDll",
    "CopyFile",
    "CreateFile",
    "DeleteFile",
    "NtDeleteFile"

## 3.移除冗余
对于超过两次以上的重复的API调用，只保留两次调用在序列中，原理来自
"Learning Malware Representation based on Execution Sequences" 和
"Deep Learning for Classification of Malware System Call Sequences"

## 4. 清洗过短的调用数据
将过短的调用数据丢弃，长度阈值：10

## 5. 统计尚存的样本归类
将剩下的样本按类划分，得到：

≥ 20 个样本的类数目: 77
≥ 15 个样本的类数目: 160
≥ 10 个样本的类数目: 209
≥ 5  个样本的类数目: 244

## 6. 按类收集数据
按数据规模，选出满足规模的类，然后根据PE文件结构选出所有文件，按类
存储