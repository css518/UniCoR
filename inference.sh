export PYTHONPATH=.
datasets=('AtCoder' 'CodeJamData' 'CSN' 'XLCoST')
modes=('nlp2code' 'code2code' 'nlp2nlp' 'nlp+code2code+code' 'remix2code')

CUDA_VISIBLE_DEVICES=0

model_path='./checkpoint'
model_name='UniCoR'

output=./result-check
cache=${output}/cache
outputs=${output}/result
csv_name=${output}/model.csv
outputs_1=${output}/result-weight
csv_name_weight=${output}/model_weight.csv
    
for dataset in "${datasets[@]}"
do
    if [ ${dataset} == "XCE" ]; then
        langs=('C' 'C++' 'C#' 'Go' 'Java' 'Javascript' 'PHP' 'Python' 'Ruby' 'Rust' 'Scala')
    else
        langs=('java' 'python')
    fi

    for query_lang in "${langs[@]}"
    do
        for candidate_lang in dataset/$dataset/*.jsonl
        do
            for mode in "${modes[@]}"
            do
                CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python script/eval.py \
                    --cache_dir ${cache} \
                    --output_dir ${outputs}\
                    --model_name_or_path ${model_path}  \
                    --tokenizer_name ${model_path} \
                    --dataset ${dataset} \
                    --query_file dataset/$dataset/${query_lang}.jsonl \
                    --candidate_file dataset/$dataset/${candidate_lang}.jsonl\
                    --code_length 256 \
                    --nl_length 128 \
                    --train_batch_size 64 \
                    --eval_batch_size 64 \
                    --learning_rate 2e-5 \
                    --model ${model_name}\
                    --mode ${mode}\
                    --seed 123456
            done

            python script/search_best.py \
                    --query1_path ${cache}/Vector-${model}-${dataset}-${query_lang}-nlp.pkl  \
                    --query2_path ${cache}/Vector-${model}-${dataset}-${query_lang}-code.pkl \
                    --target_path ${cache}/Vector-${model}-${dataset}-${candidate_lang}-code.pkl \
                    --prefix ${model}-${dataset}-nlp+code2code-${query_lang}2${candidate_lang} \
                    --output_dir ${outputs_1}  
        done
    done  
done

python script/Analysis.py --path ${outputs} --output_dir ${csv_name}
python script/Analysis_weight.py --path ${outputs_1} --output_dir ${csv_name_weight}