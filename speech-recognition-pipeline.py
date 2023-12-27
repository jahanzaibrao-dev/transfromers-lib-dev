from transformers import pipeline

transcriber = pipeline(task="automatic-speech-recognition", model="distil-whisper/distil-small.en")
demoAudio = "https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231217%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231217T093508Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6fa47c9ff132b5ffe0506af87d370c0ed5518a186448206e58246b5c5ca907d63923385795b82aa8b858421d575c26b4138d99b1c05a40321c586db2992c77ef10ee58f56dfcee5ba4e94f6b51549a31671be154a3413f3d97a168cc16acecc5be155485f9a3978df5f12f9e733be32053492f751bd69508c311c125bebc4be4a9477adfd036ab081c73de994fc647bd12dc490fa575954a28317a5811f31556a06620c09495a80860b1403533354088146c7f12e8d5c9cbdbe6e96ad63fce64023bf92116bcc3bfd3b53c12e592a9aa34d657bfd7bee72f4bb9928bc637e77d4966ecd3235696e098e413a9147e02a8236ac1635121ca48ef2cd762bbf2e1f7"

response = transcriber(demoAudio)

print(response)