# GitHub에 프로젝트 푸시 및 Streamlit Cloud 배포 안내서

이 문서는 현재 프로젝트를 GitHub에 푸시하고, Streamlit Cloud를 통해 웹 애플리케이션으로 배포하는 전체 과정을 안내합니다.

---

## 1. GitHub에 프로젝트 푸시하기

로컬 프로젝트를 GitHub 리포지토리에 올려 공유하고 관리할 수 있도록 준비하는 단계입니다.

### 1단계: `.gitignore` 파일 생성 또는 확인

**가장 중요한 단계입니다.** API 키와 같은 민감 정보, 가상 환경 폴더, 캐시 파일 등이 GitHub에 올라가지 않도록 설정해야 합니다.

프로젝트 루트 디렉터리(`.`)에 `.gitignore` 파일이 없다면 아래 내용으로 새로 만들어주세요. 이미 있다면 아래 항목들이 포함되어 있는지 확인하고 빠진 내용을 추가해주세요.

```gitignore
# Python
.venv/
__pycache__/
*.pyc

# Secrets - 절대 GitHub에 올리면 안 됩니다!
.env

# IDE/Editor specific
.idea/
.vscode/

# OS specific
.DS_Store
```

### 2단계: Git 로컬 리포지토리 초기화 및 커밋

```bash
# 1. Git 리포지토리 초기화
git init

# 2. 모든 파일을 Staging Area에 추가 ( .gitignore 제외 )
git add .

# 3. 첫 번째 커밋 작성
git commit -m "Initial commit: 프로젝트 초기 설정"
```

### 3단계: GitHub에서 새 리포지토리 생성

1.  [GitHub](https://github.com/new)에 접속하여 새 리포지토리(Repository)를 생성합니다.
2.  **Repository name**을 원하는 대로 정합니다. (예: `gemini-streamlit-app`)
3.  **Public**으로 설정해야 Streamlit Cloud 무료 티어에서 배포할 수 있습니다.
4.  **"Add a README file", "Add .gitignore", "Choose a license"는 모두 체크하지 않고** `Create repository` 버튼을 클릭합니다. (이미 로컬에서 다 만들었기 때문입니다.)

### 4단계: 로컬 리포지토리와 원격 리포지토리 연결 및 푸시

GitHub에서 리포지토리를 만들면 나오는 안내 페이지의 `…or push an existing repository from the command line` 부분을 따릅니다. 아래 명령어를 터미널에 입력하세요.

**`YourUsername`과 `YourRepositoryName`은 본인의 정보에 맞게 수정해야 합니다.**

```bash
# 1. 원격 리포지토리 주소 추가
git remote add origin https://github.com/YourUsername/YourRepositoryName.git

# 2. 기본 브랜치 이름을 main으로 설정
git branch -M main

# 3. 원격 리포지토리로 코드 푸시
git push -u origin main
```

---

## 2. Streamlit Cloud에 배포하기

GitHub에 올라간 프로젝트를 이제 Streamlit Cloud를 통해 웹 애플리케이션으로 배포합니다.

### 1단계: `requirements.txt` 파일 확인 및 업데이트

Streamlit Cloud는 이 파일을 보고 프로젝트에 필요한 라이브러리를 설치합니다. 현재 가상환경에 설치된 라이브러리 목록을 정확히 반영하도록 아래 명령어로 업데이트하는 것을 권장합니다.

```bash
# .venv/Scripts/activate 실행 후
pip freeze > requirements.txt
```

### 2단계: Streamlit Cloud 로그인 및 앱 생성

1.  [Streamlit Cloud](https://share.streamlit.io/)에 GitHub 계정으로 로그인합니다.
2.  우측 상단의 `New app` 버튼을 클릭합니다.
3.  `Deploy from an existing repo`를 선택합니다.

### 3단계: 배포 설정

-   **Repository**: 위에서 푸시한 GitHub 리포지토리 (예: `YourUsername/YourRepositoryName`)를 선택합니다.
-   **Branch**: `main`을 선택합니다.
-   **Main file path**: Streamlit 앱의 메인 파일 경로를 지정합니다. 현재 프로젝트에서는 `main.py` 또는 `utils/dashboard.py`가 될 가능성이 높습니다. 실행할 파일을 정확히 지정해주세요.
-   **App URL**: 원하는 대로 앱의 URL을 설정할 수 있습니다.

### 4단계: 민감 정보(Secrets) 설정 (가장 중요!)

로컬의 `.env` 파일에 저장했던 API 키 등은 `.gitignore` 때문에 GitHub에 올라가지 않았습니다. 따라서 Streamlit Cloud에서 직접 이 정보들을 설정해주어야 합니다.

1.  `Advanced settings...`를 클릭합니다.
2.  **Secrets** 섹션에 로컬 `.env` 파일의 내용을 그대로 복사하여 붙여넣습니다.

    예를 들어, `.env` 파일 내용이 아래와 같다면,

    ```
    OPENAI_API_KEY="sk-..."
    GOOGLE_API_KEY="AIza..."
    ```

    Secrets 입력창에도 똑같이 입력해줍니다.

### 5단계: 배포

`Deploy!` 버튼을 클릭합니다.

Streamlit Cloud가 GitHub 리포지토리에서 코드를 가져오고, `requirements.txt`를 기반으로 라이브러리를 설치한 후, 앱을 실행합니다. 이 과정은 몇 분 정도 소요될 수 있습니다. 배포가 완료되면 앱 URL로 접속하여 동작을 확인할 수 있습니다.
