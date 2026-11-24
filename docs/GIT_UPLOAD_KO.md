# GitHub 공개 저장소 업로드 절차

이 폴더는 GitHub에 올릴 파일만 추린 **저장소 작업본**입니다. 전달받은 ZIP을
그대로 GitHub 저장소 안에 한 파일로 올리지 말고, **압축을 푼 내용**을 Git으로
업로드하세요. C&G는 제출 시 공개되고 사용 설명이 있는 저장소를 요구합니다.

## Windows에서 GitHub Desktop으로 업로드하기 (처음 Git을 쓰는 경우 권장)

### 1. ZIP 압축 풀기

전달받은 `MCWP_CGeosciences_Repository.zip`을 Windows에서 다운로드한 뒤
마우스 오른쪽 → **압축 풀기/모두 추출**을 선택합니다. 예를 들어 최종 위치를
`C:\MCWP_Git\MCWP_CGeosciences_Repository\`로 정합니다. 파일 탐색기에서
이 폴더를 열어 `README.md`, `LICENSE`, `.gitignore`, `mcwp`, `datasets`,
`weights`가 바로 보이는지 확인합니다. ZIP 안에 이미 이름 붙은 폴더가 있으므로
같은 이름의 폴더가 두 겹일 수도 있습니다. **README.md가 바로 보이는 안쪽
폴더**를 뒤 단계에서 선택하세요. 전달 ZIP 자체는 이 폴더 안에 복사하지 않습니다.

### 2. 공개할 내용을 먼저 확인

`LICENSE`의 저작권자·MIT 공개 권한과 `datasets/redsea/raw/` 및
`datasets/pluto/velocity_models/` 파생 배열의 재배포 조건을 확인합니다.
연구 서버 경로가 남은 JSON도 아래 '공개 전 파일 확인' 항목을 참고하세요.
아직 확인 중이라면 아래의 **Publish repository**는 누르지 않고 로컬에
저장소를 만드는 단계까지만 진행할 수 있습니다.

### 3. GitHub Desktop에서 로컬 저장소 만들기

공식 [GitHub Desktop](https://desktop.github.com/)을 Windows에 설치하고
본인 GitHub 계정으로 로그인합니다. 상단 **File → Add local repository →
Choose...**에서 1단계의 `README.md`가 바로 보이는 폴더를 고릅니다.
"이 디렉터리는 Git 저장소가 아닙니다. 여기서 저장소를 만드시겠습니까?"라는
안내가 나오면 **create a repository**를 누릅니다. 생성 화면에서 README,
Git ignore, License 자동 생성은 선택하지 않습니다. 제공한 파일에 이미
모두 있습니다. **Create repository**를 누릅니다. 이 과정은 선택한 폴더
안에 Git 관리 정보를 추가할 뿐, 원본 연구 서버 폴더에는 영향을 주지 않습니다.

### 4. 첫 변경 기록(커밋) 만들기

GitHub Desktop 왼쪽 **Changes**에 제공 파일들이 표시됩니다. `README.md`,
`LICENSE`, `mcwp/`, `weights/`, `results/`가 있는지 확인합니다. 원래 받았던
`number1.zip`, `number2.zip`, 논문 PDF, 실험 로그와 전체 데이터가 표시되면
잘못된 폴더를 선택했으니 게시 전에 중단하고 1단계 폴더를 다시 고릅니다.
왼쪽 아래 **Summary (required)**에
`Release MCWP study code and reproducibility records`를 쓰고
**Commit to main**을 누릅니다. 화면 상단에 **Publish repository**가 보이면
첫 로컬 커밋이 만들어진 것입니다. 변경 수가 0이라면 **History**에서
첫 커밋과 제공 파일들이 실제로 들어 있는지 확인합니다.

### 5. GitHub에 공개 게시하기

**Publish repository** → 본인 **Account** 선택 → **Name**에
`mcwp-wavefield` 입력 → **Keep this code private 체크 해제** →
개인 계정이면 **Organization: None** → **Publish Repository**.
완료 후 **Repository → View on GitHub**를 열어 README와 폴더가 보이고
저장소에 `Public` 표시가 있는지 확인합니다. 이때 비로소 다른 사람이
저장소를 볼 수 있습니다. 다음의 Release 단계까지 마친 후 비로그인 창에서도
접근되는지 확인하세요.

### 6. 대형 체크포인트를 GitHub Release에 올리기

원래 전달한 `number2.zip`을 Windows에서 따로 풀어
`number2/ema_0.999_100000.pt` 파일(약 503 MB)을 찾습니다. 파일 탐색기
**보기 → 파일 확장명**을 켜고
`genwp_retrained_ema_0.999_100000.pt`로 이름을 바꿉니다.
웹브라우저에서 방금 만든 GitHub 저장소 → 오른쪽 **Releases** →
**Draft a new release** → **Choose a tag**에서 새 태그 `v1.0.0` 생성
(대상 브랜치 `main`) → **Release title**에 `MCWP study checkpoints` →
아래 파일 첨부 칸에 이 `.pt` 파일을 끌어 놓거나 선택 → 업로드 완료를
확인한 다음 **Publish release**를 누릅니다. ZIP이나 503 MB 파일을
`Changes`에 넣거나 일반 Git 커밋으로 올리지 마세요.

원한다면 Windows PowerShell에서 원본 가중치 해시를 검사할 수 있습니다.
경로는 실제로 압축을 푼 위치로 바꾸세요:

```powershell
(Get-FileHash 'C:\MCWP_Release\number2\ema_0.999_100000.pt' -Algorithm SHA256).Hash
```

기대 값은
`ceba73e1eaf379081cdbfad7f3f4a4e19696bb9cd276ad3a35474d22a6bfe6b1`입니다.
README에는 `Releases`에서 `v1.0.0`의 해당 파일을 받으라고 이미 적어
놓았으므로 이 단계에서 README나 `SHA256SUMS.txt`를 수정할 필요는 없습니다.
마지막으로 저장소와 Release를 비로그인 브라우저에서 열어 보고, 원고의
Code availability에 **실제 공개 저장소 URL**을 적습니다.

---

## 터미널에서 Git 명령으로 업로드하기 (대안)

## 1. 전달 파일을 원하는 작업 컴퓨터에 풀기

Linux/WSL 또는 GPU 서버의 작업 디렉터리에서:

```bash
unzip MCWP_CGeosciences_Repository.zip
cd MCWP_CGeosciences_Repository
```

`README.md`, `LICENSE`, `mcwp/`, `datasets/`, `weights/`가 **현재 디렉터리의
바로 아래**에 보여야 합니다. 압축 안의 `MCWP_CGeosciences_Repository/`
디렉터리까지 Git 저장소 루트가 됩니다. 원래 받은 ZIP 전부를 여기에 다시
복사하지 마세요.

## 2. 로컬 사전 검사

```bash
sha256sum -c SHA256SUMS.txt
python -m pip install numpy==1.26.4
python -m unittest discover -s tests -v
python scripts/evaluate_final66.py --groups RedSea --max-cases 2 --list
```

첫 명령은 압축 해제 및 파일 손상을 확인합니다. NumPy가 이미 있다면 설치
명령을 생략할 수 있습니다. 테스트와 `--list`는 CUDA를 사용하지 않습니다.
PyTorch/CUDA 모델 전체 추론은 제작 환경에 없어서 수행하지 못했습니다.
원래 사용한 서버 환경에서 아래 GPU 시험을 추가로 진행하세요.

```bash
python datasets/redsea/generate_redsea_cases.py \
  --models 1 --out-dir GenWP-pub/dataset/redsea_ood_range_matched/outputs \
  --device cuda:0
python scripts/evaluate_final66.py --data-root GenWP-pub \
  --groups RedSea --max-cases 1 --models long100 --device cuda:0
```

FD 사례 파일이 이미 있는 서버에서는 생성 명령이 덮어쓰기를 거부합니다.
기존 사례에 대해서는 먼저 `--verify-inputs-only`로 해시를 검증하세요.
새로 만든 사례의 해시가 다르면 GPU/PyTorch/FD 버전을 확인하고 원 논문
수치와 동일하다고 주장하지 마세요. `README.md`와 `docs/USER_GUIDE.md`에
데이터 경로 및 평가 조건이 설명돼 있습니다.

## 3. 공개 전 파일 확인

- `LICENSE`의 MIT 선택과 권리자 표기, 공동저자·기관의 공개 허용 여부를 확인합니다.
- `datasets/redsea/raw/*.npy`와 `datasets/pluto/velocity_models/*.npz` 같은
  원본 자료 파생 배열의 공개 재배포 조건을 확인합니다. 허용되지 않으면 해당
  파일을 제거하고 `docs/DATA_SOURCES.md`에 접근·재생성 절차와 한계를 적습니다.
- 개인 경로가 들어간 `provenance/`, `benchmarks/cache/config.json`,
  `results/cache/protocol.json`, `results/redsea_sensitivity/`의 프로토콜과
  사례 목록도 확인합니다. 연구 서버 경로가 재현 기록으로 남아 있습니다.
  비밀키·비밀번호가 포함되지 않았는지 공개 전에 직접 확인하세요.
- `weights/`의 다섯 파일은 각각 GitHub의 일반 Git 단일 파일 제한보다
  작습니다. `number2.zip`의 503 MB GenWP-Retrained 체크포인트는 이 Git
  폴더에 넣지 말고 **Release 자산**으로 올립니다. GenWP-Published의 원본
  가중치는 upstream GenWP 배포 위치를 연결하고 해시를 확인합니다.

파일을 수정하거나 제거했다면 아래 Git 초기화 후 `git add .`와
`python scripts/update_checksums.py`를 실행해 `SHA256SUMS.txt`를 갱신하세요.
이 스크립트는 Git에 등록한 파일만 대상으로 하므로 생성된 실험 결과는 제외합니다.

## 4. GitHub에서 빈 공개 저장소 만들기

GitHub 로그인 → 우상단 `+` → **New repository** → 예를 들어
`mcwp-wavefield` 입력 → **Public** 선택 → 초기 README, `.gitignore`,
License 자동 생성 옵션은 **체크하지 않고** → Create repository.
이미 완성된 이 폴더에 세 파일이 있기 때문입니다.

## 5. 폴더 내용을 Git으로 첫 업로드

아래 명령에서 `YOUR_USERNAME`과 저장소 이름을 실제 값으로 바꾸세요.

```bash
cd /PATH/TO/MCWP_CGeosciences_Repository
git init -b main
git add .
python scripts/update_checksums.py
sha256sum -c SHA256SUMS.txt
git add SHA256SUMS.txt
git status --short
git diff --cached --stat
git -c core.whitespace=cr-at-eol diff --cached --check
git commit -m "Release MCWP study code and reproducibility records"
git remote add origin https://github.com/YOUR_USERNAME/mcwp-wavefield.git
git push -u origin main
```

Git 신원 미설정 오류가 나오면, 이 폴더에서만 적용되게 설정하세요:

```bash
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

GitHub 로그인에는 브라우저 또는 GitHub CLI(`gh auth login`) 인증을
사용하세요. 계정 비밀번호를 터미널의 Git 인증란에 그대로 넣는 방식은
권장하지 않습니다. `git status --short` 결과에 대형 원본 데이터나 실험
로그가 들어 있으면 커밋 전 제외하세요. 원래 받은 `number1.zip`,
`number2.zip`, 전체 `Marmousi_Vmax5000_Evaluation.zip`도 Git에 추가하지
마세요.

## 6. 503 MB GenWP-Retrained 가중치를 Release 자산으로 등록

원래 전달한 `number2.zip`에서 체크포인트를 압축 해제합니다. 이것은 Git에
커밋하지 않습니다.

```bash
unzip -p /PATH/TO/number2.zip number2/ema_0.999_100000.pt \
  > ../genwp_retrained_ema_0.999_100000.pt
sha256sum ../genwp_retrained_ema_0.999_100000.pt
```

기대 해시:
`ceba73e1eaf379081cdbfad7f3f4a4e19696bb9cd276ad3a35474d22a6bfe6b1`.
GitHub 저장소의 **Releases → Draft a new release**에서 태그 `v1.0.0`,
제목 `MCWP study checkpoints`를 정하고 이 `.pt` 파일을 자산으로 첨부해
공개합니다. GitHub CLI가 설치되어 있으면 다음과 같이 할 수도 있습니다:

```bash
gh release create v1.0.0 ../genwp_retrained_ema_0.999_100000.pt \
  --title "MCWP study checkpoints" \
  --notes "GenWP-Retrained EMA 100000; verify SHA256 in weights/manifest.json"
```

GitHub는 일반 Git 파일을 100 MiB에서 차단하지만, Release의 개별 자산은
2 GiB 미만까지 허용합니다. 따라서 이 파일은 Release에 둡니다.

## 7. 공개 링크 연결 및 최종 확인

README에는 Releases의 `v1.0.0` 파일을 받는 방법이 이미 적혀 있습니다.
GenWP-Published 가중치는 공식 GenWP 저장소의 배포 링크와
기대 해시 `7f174ef7501ec6f15b457be3801e4b34b2390710f613521bc07d8082a58aecea`
를 안내합니다. GitHub에서 **로그아웃한 브라우저**로 공개 저장소·README·
Release 파일에 접근 가능한지 확인하세요.
마지막으로 원고의 Code availability에 실제 공개 저장소
URL을 넣습니다. 단일 ZIP 링크만 적는 방식은 C&G 저장소 요건을 충족하지
않습니다. 온라인으로 공개하는 행위와 라이선스 확정은 저자 측 최종 결정입니다.
