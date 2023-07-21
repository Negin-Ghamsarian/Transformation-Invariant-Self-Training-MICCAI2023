
class import_config():
    def __init__(self):
        print('import_var initialized')

    def execute(module_name):
        Categories = module_name.Categories
        Learning_Rates_init = module_name.Learning_Rates_init
        epochs = module_name.epochs
        batch_size = module_name.batch_size
        size = module_name.size
        Dataset_Path_Train = module_name.Dataset_Path_Train
        Dataset_Path_SemiTrain = module_name.Dataset_Path_SemiTrain
        Dataset_Path_Test = module_name.Dataset_Path_Test
        mask_folder = module_name.mask_folder
        Results_path = module_name.Results_path
        Visualization_path = module_name.Visualization_path
        Checkpoint_path = module_name.Checkpoint_path
        CSV_path = module_name.CSV_path
        project_name = module_name.project_name
        load = module_name.load
        load_path = module_name.load_path
        net_name = module_name.net_name
        test_per_epoch = module_name.test_per_epoch
        Net1 = module_name.Net1
        hard_label_thr = module_name.hard_label_thr
        ensemble_batch_size = module_name.ensemble_batch_size
        SemiSup_initial_epoch = module_name.SemiSup_initial_epoch
        image_transforms = module_name.image_transforms 
        affine = module_name.affine
        affine_transforms = module_name.affine_transforms 
        LW = module_name.LW
        EMA_decay = module_name.EMA_decay
        Alpha = module_name.Alpha
        strategy = module_name.strategy
        GCC = module_name.GCC
        supervised_share = module_name.supervised_share

        return Categories, Learning_Rates_init, epochs, batch_size, size,\
             Dataset_Path_Train, Dataset_Path_SemiTrain, Dataset_Path_Test,\
                  mask_folder, Results_path, Visualization_path,\
                 CSV_path, project_name, load, load_path, net_name,\
                      test_per_epoch, Checkpoint_path, Net1,\
                         hard_label_thr, ensemble_batch_size, SemiSup_initial_epoch,\
                             image_transforms, affine, affine_transforms, LW,\
                                 EMA_decay, Alpha, strategy, GCC, supervised_share
        
           