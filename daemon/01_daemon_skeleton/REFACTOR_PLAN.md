# Refactor Plan

## 1. 현재 변경 확인

- `Event` / `EventQueue` 구조가 다시 중심이 되었다.
- `Event` 안에 `request_id`, `channel_name`이 직접 들어 있어 request와 event의 경계가 섞여 있다.
- `RequestState`가 `Event request`를 들고 있어 daemon 내부 request state와 event message가 섞여 있다.
- `ChannelState`가 input/output buffer뿐 아니라 request completion 상태와 SG sequence path까지 들고 있다.
- `ExamClient::wait()`은 channel 내부 completion flag/condvar에 의존한다.
- daemon은 invalid request를 막기 위해 `Channel::complete_request()`를 호출한다.

## 2. 설계 방향

### Event와 Request 분리

- `Request`
  - client가 제출한 실제 request 정보만 가진다.
  - `request_id`, `channel_name`을 가진다.

- `Event`
  - daemon이 처리하는 queue message다.
  - register, submit, SG complete, request complete를 구분한다.
  - request 관련 event는 내부에 `Request`를 포함한다.
  - register event는 SG sequence path를 포함한다.

### Channel 역할 축소

- `Channel`은 input/output payload data window로만 사용한다.
- completion flag, completion condvar, request id generation, SG sequence path는 channel에서 제거한다.
- channel에는 payload buffer metadata만 남긴다.

### Client completion 분리

- process wait을 깨우기 위한 상태는 새 `ClientControlState`로 분리한다.
- daemon은 request complete 시 channel이 아니라 client control을 signal한다.
- client는 `wait()`에서 client control condvar를 기다린다.

### Daemon 자료구조 정리

- `ClientState`
  - registered client metadata
  - `channel_name`, `sg_sequence_file_path`, `sg_sequence`

- `RequestState`
  - active request execution state
  - `Request`, `sg_sequence`, `next_sg_index`

- `Subgraph`
  - 실행할 SG 하나와 해당 request를 가진다.
  - `Event`가 아니라 `Request`를 저장한다.

## 3. 구현 순서

1. `Request` datatype 추가.
2. `Event`를 `Request` 포함 message로 정리.
3. `ClientControlState` 추가.
4. `ChannelState`에서 completion/path/request id 관련 상태 제거.
5. `ExamClient`를 request 생성, data write, event post, client control wait 구조로 변경.
6. `ExamDaemon`을 event 처리와 request state 관리가 분리되도록 변경.
7. `Worker`/`Subgraph`를 `Request` 기반으로 변경.
8. 작업 주석 제거 및 빌드/통합 실행 확인.
