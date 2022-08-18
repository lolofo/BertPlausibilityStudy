conda activate nlp

$GIT_REP = "C:\Users\loicf\Documents\IRISA\BertPlausibilityStudy"
cd $GIT_REP

python .\inference_scripts\e_snli\regu_study.py --batch_size 4 --reg_mul 0.0
python .\inference_scripts\e_snli\regu_study.py --batch_size 4 --reg_mul 0.005
python .\inference_scripts\e_snli\regu_study.py --batch_size 4 --reg_mul 0.05
python .\inference_scripts\e_snli\regu_study.py --batch_size 4 --reg_mul 0.5