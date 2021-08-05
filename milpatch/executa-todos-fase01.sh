python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold1/test/400X/" --model "../BKPmilpatch/models/fold1/pesos-fase01.h5" --fold "fold1-fase1" >> cancer-fold1-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold2/test/400X/" --model "../BKPmilpatch/models/fold2/pesos-fase01.h5" --fold "fold2-fase1" >> cancer-fold2-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold3/test/400X/" --model "../BKPmilpatch/models/fold3/pesos-fase01.h5" --fold "fold3-fase1" >> cancer-fold3-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold4/test/400X/" --model "../BKPmilpatch/models/fold4/pesos-fase01.h5" --fold "fold4-fase1" >> cancer-fold4-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold5/test/400X/" --model "../BKPmilpatch/models/fold5/pesos-fase01.h5" --fold "fold5-fase1" >> cancer-fold5-fase01.out

python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test1.txt" --base especies --model "../milespecies/models/fold1/delta-02/pesos-fase01.h5" --fold "fold1-fase1" >> especies-fold1-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test2.txt" --base especies --model "../milespecies/models/fold2/delta-03/pesos-fase01.h5" --fold "fold2-fase1" >> especies-fold2-fase01.out
python fase01_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test3.txt" --base especies --model "../milespecies/models/pesos-fase01.h5" --fold "fold3-fase1" >> especies-fold3-fase01.out

