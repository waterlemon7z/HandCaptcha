## GUI Integration
- Author: MinSeok Choi
- Role: 본 프로그램의 뼈대가 되는 gui를 구현하고 팀원들이 구현한 각 클래스들을 연결하여 매끄럽게 작동되도록 합니다.
### Structure
- gui/controller.py - 각 클래스 및 변수로부터 상태를 받아와 GUI에 업데이트 합니다.
- gui/handcaptcha.py - GUI의 구조 및 스타일을 정의합니다.
- gui/handcaptcha.ui - qt designer 프로그램의 output입니다.
- gui/signal.py - GUI의 각 elements에서 발생한 클릭과 같은 이벤트를 받아 처리합니다.