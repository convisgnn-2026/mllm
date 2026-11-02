# 저장소 구조와 정리 기록

## 현재 연구

| 경로 | 역할 |
|---|---|
| bench/hetero/ | 현재 실측 harness·프로파일·분석·그림 코드 |
| results/hetero/ | 현재 원시 측정·파생 분석·그림 |
| docs/ | 현재 미팅·실험 계획·환경·폴더 안내 |
| env/ | 의존성 기록 |
| papers/ | 참고 논문과 외부 코드 |
| presentations/ | 발표 자료와 생성 코드; slides/에 이전 ppt 폴더 내용 |

현재 혼재 측정과 solver 필요성 분석 경로는 유지했다.

## 이전 연구

이전 `solver/`, `hetsim/`, `scripts/`, `slo_configs/`, `figures/`, `bench/`의 MIG harness, `results/`의 비 hetero 결과, 과거 문서·assets를 **archive/legacy/**로 모았다. 코드·설정·결과의 상대 관계를 유지했다. [legacy 안내](../archive/legacy/README.md)를 참고한다.

`docs/reseach_env.md`는 `docs/research_env.md`로 오타를 수정했다. 미팅 파일 `docs/1013_meeting.md`는 유지했다. 발표 파일은 presentations/로 모았다.

이동 목록 전체는 [reorganization.json](reorganization.json)에 기록했다. Markdown 상대 링크를 새 위치로 수정했고 이전 실행 스크립트의 저장소 진입 경로를 수정했다. .gitignore의 이전 결과 경로도 갱신했다. 원시 데이터와 사용자 수정 파일은 내용 손실 없이 이동했다.

## 제거한 파일

참조 검색에서 쓰이지 않는 발표 슬라이드 JPG 미리보기와 재생성 가능한 Python cache를 제거했다. 원본 발표 자료는 남겼다.

| 제거 경로 | 크기(bytes) |
|---|---:|
| docs/ppt/ong-1.jpg | 83674 |
| docs/ppt/ong-2.jpg | 88150 |
| docs/ppt/ong-3.jpg | 93096 |
| docs/ppt/ong-4.jpg | 86398 |
| docs/ppt/slide-1.jpg | 121393 |
| docs/ppt/tech-1.jpg | 108178 |
| docs/ppt/tech-2.jpg | 121012 |
| docs/ppt/tech-3.jpg | 111062 |
| docs/ppt/tech-4.jpg | 94250 |
| docs/ppt/v04-01.jpg | 25443 |
| docs/ppt/v04-02.jpg | 70058 |
| docs/ppt/v04-03.jpg | 74101 |
| docs/ppt/v04-04.jpg | 73880 |
| docs/ppt/v04-05.jpg | 64743 |
| docs/ppt/v04-06.jpg | 66066 |
| docs/ppt/v04-07.jpg | 79282 |
| docs/ppt/v04-08.jpg | 84086 |
| docs/ppt/v04-09.jpg | 87032 |
| docs/ppt/v04-10.jpg | 80311 |
| docs/ppt/v04-11.jpg | 67088 |
| bench/hetero/__pycache__ | 132146 |

합계 1,811,449 bytes 제거.

## 앞으로의 규칙

- 신규 실험 코드는 bench/hetero/, 결과는 results/hetero/<실험명>/에 둔다.
- 현재 문서는 docs/, 발표 원본은 presentations/, 종료된 연구는 archive/legacy/에 둔다.
- papers/의 외부 코드는 폴더 구조를 유지한다.
- 원시 데이터와 manifest를 보존하고 요약·그림은 재생성한다.
- 기존 iso_cost와 SLO plot은 과거 탐색 가정으로 읽으며 새 서비스 요구로 자동 채택하지 않는다.
