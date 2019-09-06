DATA PREPROCESSING --MSRVTT dataset
 C3D feature extraction -- fc6 features 

--prepro_vocab.py 
 ----input : msr_vtt videoinfo json,'data/test_videodatainfo.json
 ----output :builds vocabulary ,video to caption dictionary, info files

--dataloader.py
    --inputs : vocab,info files from above
    --output : when called data is loaded in batches
    
--opt.py
-----change the parameters of the 


---To train
Python tarin.py
----change opt.py
-----output :learnt model.pth

--To test
python test.py
-----input  : give trained model.path
-----output : scores