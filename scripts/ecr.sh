aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 381491870028.dkr.ecr.us-east-2.amazonaws.com
aws ecr create-repository --repository-name shadow-trainer


docker tag shadow-trainer:latest 381491870028.dkr.ecr.us-east-2.amazonaws.com/shadow-trainer:latest
docker push 381491870028.dkr.ecr.us-east-2.amazonaws.com/shadow-trainer:latest



# {
#     "repository": {
#         "repositoryArn": "arn:aws:ecr:us-east-2:381491870028:repository/shadow-trainer",
#         "registryId": "381491870028",
#         "repositoryName": "shadow-trainer",
#         "repositoryUri": "381491870028.dkr.ecr.us-east-2.amazonaws.com/shadow-trainer",
#         "createdAt": "2025-06-30T18:17:29.885000-04:00",
#         "imageTagMutability": "MUTABLE",
#         "imageScanningConfiguration": {
#             "scanOnPush": false
#         },
#         "encryptionConfiguration": {
#             "encryptionType": "AES256"
#         }
#     }
# }
# (END)