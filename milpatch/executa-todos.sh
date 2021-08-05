#python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold1/test/400X/" --model "../BKPmilpatch/models/fold1/pesos-fase02.h5" --fold "fold1" >> cancer-fold5-fase01.out
#python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold2/test/400X/" --model "../BKPmilpatch/models/fold2/pesos-fase02.h5" --fold "fold2" >> cancer-fold2-fase02.out
#python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold3/test/400X/" --model "../BKPmilpatch/models/fold3/pesos-fase02.h5" --fold "fold3" >> cancer-fold3-fase02.out
#python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold4/test/400X/" --model "../BKPmilpatch/models/fold4/pesos-fase02.h5" --fold "fold4" >> cancer-fold4-fase02.out
#python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/cancer/fold5/test/400X/" --model "../BKPmilpatch/models/fold5/pesos-fase02.h5" --fold "fold5" >> cancer-fold5-fase02.out

python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test1.txt" --base especies --model "../milespecies/models/fold1/delta-02/pesos-fase02.h5" --fold "fold1" >> especies-fold1-fase02.out
python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test2.txt" --base especies --model "../milespecies/models/fold2/delta-03/pesos-fase02.h5" --fold "fold2" >> tee especies-fold2-fase02.out
python fase02_clfpatch_ptx64.py --image_dir "/home/willian/bases/especies/" --labels "/home/willian/bases/especies/labels/test3.txt" --base especies --model "../milespecies/models/pesos-fase02.h5" --fold "fold3" >> tee especies-fold3-fase02.out

