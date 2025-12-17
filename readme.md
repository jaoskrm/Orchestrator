# 1) Clean reset (wipe files from previous task)
docker exec agent-sandbox bash -lc "rm -rf /work/*"

# 2) Copy the new task into the container
docker cp .\runs\task_001\. agent-sandbox:/work

# 3) Run tests
docker exec agent-sandbox python -m pytest -q

# 4) Clean up
docker stop agent-sandbox
docker rm agent-sandbox


#  PER TASK COMMANDS
docker exec agent-sandbox sh -lc "rm -rf /work/task/*"
docker cp .\runs\task_001\. agent-sandbox:/work/task
docker exec agent-sandbox sh -lc "cd /work/task && python -m pytest -q -p no:cacheprovider"
