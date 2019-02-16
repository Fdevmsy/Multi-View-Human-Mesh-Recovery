## README

### pose estimation multi-view

```

```



1. Convert video to image:

   ```shell
   sudo apt install ffmpeg
   # then put video_to_image.sh together with your /mpi_ing_3dhp/ folder, which contains the mpi_ing_3dhp dataset 
   ./video_to_image.sh
   ```

   

2. Create tf_records for mpi_ing_3dhp dataset. 

   ```shell
   cd path/to/hmr/
   python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory /path/to/mpi_inf_3dhp --output_directory /path/to/tf_records/mpi_inf_3dhp
   
   # in my example, all original data is stored in hmr/all_data, all converted tf recoreds are stored in hmr/tf_datasets
   python -m src.datasets.mpi_inf_3dhp_to_tfrecords --data_directory ./all_data/mpi_inf_3dhp --output_directory ./tf_datasets/mpi_inf_3dhp
   
   ```

3. Train the model

   ```shell
   cd hmr/
   ./do_train.sh
   ```

   