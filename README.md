# 6th_paper
* low level 로 접근하지않는 쪽으로 연구진행
* 현재 코딩하는중
* edge detector는 나중
* **현재 다른 버전들을 본컴, 서브컴 및 코랩에 각각 실험하고있는중**
* 아이디어 하나 더 추가, 지금은 5번째 논문 거의 완성단계라 실험 중단
* 이미지와 라벨과 모델을 모두 다시 수정하였음
<br/>

* 현재 3버전에 대해 각각 본컴, 서브컴 및 코랩에 돌리고있음 (이들의 테스트방법을 각 영역에 대해 개별적으로 한 뒤 concat하는 방식으로 제안해보자)
(이건 실험이 끝나는데로 바로 해볼것)
* Current - writing review comment (Done)
* **train_7_main_com.py is promising (0.5951)**
* colab 및 서브컴에 돌리고있는 train_7_colab.py 및 train_7_sub_com.py 꼭 결과보기
<br/>

* model_9_fix는 코랩에 re학습중 (change focal to cross entropy)
* train_9_fix.py sub com
* train_9_fix2.py main com

* 최종 성능 ==> test IoU: 0.6716, test F1_score: 0.8035, test sensitivity: 0.8350, test precision: 0.7744 (코랩에서 실험한것임 기억해!!!!)train_9_fix2.py

* model_12.py 아직까지는 괜찮음 --> 본컴에 실험중 (**기억해**)
* model_13.py 는 코랩으로 실험중 (**기억해**), 오늘 저녁에 896 해상도 및 본컴에서 응용한것을 이 파일에 그대로 적용해보고 실험 해보자!!!!!!!!!!!!!!!!!!!!!!
* 현재 코랩에서는 model_12에서 조금 수정한 버전을 실험중임  (**기억해**)
