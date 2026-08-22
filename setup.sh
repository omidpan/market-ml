#!/bin/zsh

# Color definitions for terminal output
RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
NC='\e[0m' # No Color

# 1. Initialize variables
IBG_PORT=""
MLF_PORT=""
API_PORT=""

# Get the absolute path of the directory where this script lives
SCRIPT_DIR="${0:A:h}"
ENV_FILE="$SCRIPT_DIR/.env"
OLD_ENV_FILE="$SCRIPT_DIR/.env.old"

# 2. Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -ibg)
            IBG_PORT="$2"
            shift 2
            ;;
        -mlf)
            MLF_PORT="$2"
            shift 2
            ;;
        -api)
            API_PORT="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Error: Unknown argument: $1${NC}"
            echo "Usage: $0 -ibg <port> -mlf <port> -api <port>"
            exit 1
            ;;
    esac
done

# 3. Validate inputs
if [[ -z "$IBG_PORT" || -z "$MLF_PORT" || -z "$API_PORT" ]]; then
    echo -e "${RED}Error: Missing required ports.${NC}"
    echo "Usage: $0 -ibg <port> -mlf <port> -api <port>"
    exit 1
fi

# 4. Ensure directories exist
BASE_DIR="${PROJECT_HOME:-$HOME}"
mkdir -p "$BASE_DIR/stp/data"
echo -e "${GREEN}Success: Directories ensured at: $BASE_DIR/stp/data${NC}"

# 5. Handle .env backup and renewal
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Notice: Existing .env found. Moving to $OLD_ENV_FILE${NC}"
    mv "$ENV_FILE" "$OLD_ENV_FILE"
fi

echo -e "${BLUE}Info: Generating new .env file...${NC}"

# 6. Write resolved values into the new .env file
cat << EOF > "$ENV_FILE"
####### data path
DATA_PATH=$BASE_DIR/data
IBG_RAW_PATH=$BASE_DIR/data
#### config path
CONFIG_PATH=$BASE_DIR/config
## project path 
REPORTS_PATH=$BASE_DIR/reports
##### scraper(IB GateWay proxy)
HOST=127.0.0.1
PORT=$IBG_PORT


######### MLFLOW variables ########

###### MLFLOW PORTS ################

MLFLOW_HOST_PORT=$MLF_PORT
MLFLOW_INTERNAL_PORT=$MLF_PORT

TRACKER_HOST_PORT=$API_PORT
TRACKER_INTERNAL_PORT=$API_PORT

TRACKER_API_TIMEOUT_SECONDS=120
##########PATHs #############
MLFLOW_RUNTIME_HOST_PATH=$BASE_DIR/mlflow
MLFLOW_DB_CONTAINER_PATH=/mlflow/db
MLFLOW_ARTIFACTS_CONTAINER_PATH=/mlflow/artifacts
MODEL_RUNS_HOST_PATH=$SCRIPT_DIR/data/models
MODEL_RUNS_CONTAINER_PATH=/model-runs
TRACKER_API_BASE_URL=http://localhost:$API_PORT
# Paths inside both containers
MLFLOW_DATABASE_PATH=/mlflow/db/mlflow.db
MLFLOW_ARTIFACTS_PATH=/mlflow/artifacts
REPO=$SCRIPT_DIR
COLAB_TOKEN= your token
EOF

# 7. Display completion & action notice
echo -e "${GREEN}Setup completed successfully:${NC}"
echo "  -> IBG Port:      $IBG_PORT"
echo "  -> MLflow Port:   $MLF_PORT"
echo "  -> Tracker API:   $API_PORT"
echo -e "  -> Created file:  ${BLUE}$ENV_FILE${NC}"
echo -e "${YELLOW}ACTION REQUIRED: Update COLAB_TOKEN in $ENV_FILE if needed.${NC}\n"

# 8. Source the .env file
echo -e "${BLUE}Info: Sourcing $ENV_FILE...${NC}"
set -a
source "$ENV_FILE"
set +a
echo -e "${GREEN}Environment variables exported successfully.${NC}\n"

# 9. Prompt to run docker compose
read -r "REPLY?Do you want to run 'docker compose up -d' now? [Y/n]: "
if [[ "$REPLY" =~ ^[Yy]?$ ]]; then
    echo -e "${BLUE}Info: Starting Docker containers...${NC}"
    docker compose up -d
else
    echo -e "${YELLOW}Skipped Docker Compose startup.${NC}"
fi