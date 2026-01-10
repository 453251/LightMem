# An example of running the baseline model. 
# Please modify the following variables to fit your own dataset and memory configuration. 
# ========================================================
memory_type="MemZero"
dataset_type="MobileBench"
llm_model_type="gpt"
dataset_path="/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/datasets/mobilebench_data.json"
# dataset_path="/disk/disk_4T_2/chenyijun/A-mem/data/locomo10.json"
config_path="/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/configs/MemZero.json"
num_workers=1
tokenizer_path="gpt-4o-mini"
log_dir="MemZero_${llm_model_type}_MobileBench_analyze_logs"
token_cost_prefix="token_cost"
# pid_prefix="process"
# pid_prefix="process_eval"
pid_prefix="process_analyze"
search_results_file="MemZero_gpt-4o-mini_MobileBench_30_0_2.json"
evaluation_results_file="MemZero_gpt-4o-mini_MobileBench_30_0_2_evaluation.json"
ranges=(
    "0 2"
)
api_keys=(
    # "sk-ZRpF8xO5LyO8Y0eckr8RncRmW1jlkfrRmEV5ERJv6shvzv5g"
    "sk-yr563lndTwVVDft9kSoR9CPuYYZxYFSsS1jAzrmbhTfiiqD5"
    # "sk-g05bl7t1V6aeu19QkxQOH77dgQQEiACGoJTUUIXmaBlqGrg2"
    # "sk-z6VMJp6p8zR1IBfXLZLiSGwPKISV84xKgozQvbhGU7AwWR7X"
    # "sk-U4Ct0bE9RXpSLOV3kLAfO8W4X3TWxMjJFvsawPeOCDKqQTWr"
    # "sk-Z3JMX3A0afIy4b6uS9YZ49eXPJxlvVuFWgaiqO40pW0DDXZw"
    # "sk-RbEjV9qrdLGmwgnJdf6lPvqxJso9f95r6BGaGP2eANuVCnci"
    # "sk-COB3cTwcj65YUYBuPkoZMDdcmfy9J5trvrzPJtfFjjvNJYKx"

)
base_urls=(
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
)
api_keys_for_image=(
    # "sk-ZRpF8xO5LyO8Y0eckr8RncRmW1jlkfrRmEV5ERJv6shvzv5g"
    "sk-yr563lndTwVVDft9kSoR9CPuYYZxYFSsS1jAzrmbhTfiiqD5"
    # "sk-g05bl7t1V6aeu19QkxQOH77dgQQEiACGoJTUUIXmaBlqGrg2"
    # "sk-z6VMJp6p8zR1IBfXLZLiSGwPKISV84xKgozQvbhGU7AwWR7X"
    # "sk-U4Ct0bE9RXpSLOV3kLAfO8W4X3TWxMjJFvsawPeOCDKqQTWr"
    # "sk-Z3JMX3A0afIy4b6uS9YZ49eXPJxlvVuFWgaiqO40pW0DDXZw"
    # "sk-RbEjV9qrdLGmwgnJdf6lPvqxJso9f95r6BGaGP2eANuVCnci"
    # "sk-COB3cTwcj65YUYBuPkoZMDdcmfy9J5trvrzPJtfFjjvNJYKx"

)
base_urls_for_image=(
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
    "https://www.DMXapi.com/v1"
)
# ========================================================

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

for ((i=0; i<${#ranges[@]}; i++)); do
    read start_idx end_idx <<< "${ranges[$i]}"

    # ========== 创建独立的 MEM0_DIR ==========
    # ========== 仅 MemZeroGraph / MemZero 才设置 MEM0_DIR ==========
    if [[ "$memory_type" == "MemZeroGraph" || "$memory_type" == "MemZero" || "$memory_type" == "NaiveRAG" ]]; then
        mem0_worker_dir="/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/mem0_worker_dir_${memory_type,,}_${llm_model_type}/mem0_worker_$((i+1))"
        mkdir -p "$mem0_worker_dir"
        export MEM0_DIR="$mem0_worker_dir"
    else
        unset MEM0_DIR
    fi

    export HF_ENDPOINT="https://hf-mirror.com"
    export OPENAI_API_KEY="${api_keys[$i]}" 
    export OPENAI_API_BASE="${base_urls[$i]}"
    export OPENAI_API_KEY_FOR_IMAGE="${api_keys_for_image[$i]}" 
    export OPENAI_API_BASE_FOR_IMAGE="${base_urls_for_image[$i]}"

    log_file="${log_dir}/${pid_prefix}_$((i+1))_${start_idx}_${end_idx}.log"
    token_cost_file="${token_cost_prefix}_${memory_type,,}_${llm_model_type}_$((i+1))_${start_idx}_${end_idx}"
    pid_file="${log_dir}/${pid_prefix}_$((i+1)).pid"

    [ ! -f "$log_file" ] && touch "$log_file"
    # nohup python memory_construction.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --token-cost-save-filename "$token_cost_file" \
    #     --tokenizer-path "$tokenizer_path" \
    #     --rerun \
    #     --message-preprocessor "memories.datasets.locomo_preprocessor:NaiveRAG_style_message_for_LoCoMo" \
    #     > "$log_file" 2>&1 &

    # nohup python memory_construction.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --token-cost-save-filename "$token_cost_file" \
    #     --tokenizer-path "$tokenizer_path" \
    #     --rerun \
    #     > "$log_file" 2>&1 &

    # nohup python memory_search.py \
    #     --memory-type "$memory_type" \
    #     --dataset-type "$dataset_type" \
    #     --dataset-path "$dataset_path" \
    #     --config-path "$config_path" \
    #     --num-workers "$num_workers" \
    #     --start-idx "$start_idx" \
    #     --end-idx "$end_idx" \
    #     --strict \
    #     --top-k 30 \
    #     > "$log_file" 2>&1 &
    
    # nohup python memory_evaluation.py \
    #     --search-results-path "$search_results_file" \
    #     --qa-model "gpt-4o-mini" \
    #     --judge-model "gpt-4o-mini" \
    #     --qa-batch-size 8 \
    #     --judge-batch-size 8 \
    #     --api-config-path "/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/configs/api_qwen.json" \
    #     > "$log_file" 2>&1 &

    nohup python error_attribution.py \
        --evaluation-results-path "$evaluation_results_file" \
        --memory-type "$memory_type" \
        --memory-config-path "$config_path" \
        --judge-model "gpt-4o-mini" \
        --top-k 20 \
        --batch-size 8 \
        --num-workers 1 \
        --api-config-path "/disk/disk_4T_2/chenyijun/LightMem/src/lightmem/memory_toolkits/memories/configs/api_qwen.json" \
        > "$log_file" 2>&1 &

    echo $! > "$pid_file"
    sleep 10
done