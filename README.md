![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge\&logo=windows\&logoColor=white)  

![coverage badge](./coverage.svg)  

# Scheduling Problem
使用深度強化學習求解動態機台數量的彈性零工式排程問題。  
Learning to Dispatch for Flexible Job-Shop Scheduling with Dynamic Number of Machines via Deep Reinforcement Learning.

---
## 目錄結構 Directory Structure
=<font color='#ff3'>= 待寫 =</font>=
```
├── a
│   ├── b
│   └── c
└── d
```

## 開發環境 Development Environment
- Python 版本為 3.10.8
- 推薦 IDE：PyCharm  

操作說明以 Windows 作業系統作為範例，Linux 或 macOS 要注意輸入指令會有不同。  

首先，您可以使用 conda 指令來建立一個虛擬開發環境。
```bash
conda create --name scheduling_problem python=3.10
conda activate scheduling_problem
```
下一步，安裝本專案所需的相依套件。您可以在根目錄使用下列指令進行安裝：
```bash
pip install -r requirements.txt
```

## 單元測試 Unit Test
本專案使用 pytest 進行單元測試，使用 coverage.py 收集測試覆蓋率。  
下列指令將運行所有測試，並生成覆蓋率數據。
```bash
coverage run --omit="*test.py" -m pytest
```
下列指令將覆蓋率數據轉換成 html 的報表。  
您可以開啟 `./tests/coverage/index.html` 來檢視程式碼覆蓋狀況。  
```bash
coverage html --skip-empty -d .\tests\coverage
```
下列指令將生成覆蓋率徽章（badge）。  
```bash
coverage-badge -o coverage.svg -f
```
您若是在 Windows 作業環境下運行，可以參考 `coverage.ps1` 檔案的指令。

目前尚未利用 GitHub Action 自動執行測試腳本。 因此當版本更新時，需手動產生覆蓋率數據。  
若想使用 GitHub Action，請參考 https://github.com/tj-actions/coverage-badge-py

## 操作介紹 Instruction
=<font color='#ff3'>= 待寫 =</font>=

## 錯誤報告 Bug Report
運行本專案時遇到任何問題，可以透過以下方式聯絡作者：
- email: hung61601@gmail.com
