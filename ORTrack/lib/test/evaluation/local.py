from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.biodrone_path = 'F:\Cornea\data\biodrone'
    settings.davis_dir = ''
    settings.dtb70_path = 'F:\Cornea\data\dtb70'
    settings.got10k_lmdb_path = 'F:\Cornea\data\got10k_lmdb'
    settings.got10k_path = 'F:\Cornea\data\got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = 'F:\Cornea\data\itb'
    settings.lasot_extension_subset_path_path = 'F:\Cornea\data\lasot_extension_subset'
    settings.lasot_lmdb_path = 'F:\Cornea\data\lasot_lmdb'
    settings.lasot_path = 'F:\Cornea\data\lasot'
    settings.network_path = 'F:\Cornea\ORTrack\test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'F:\Cornea\data\nfs'
    settings.otb_path = 'F:\Cornea\data\otb'
    settings.prj_dir = 'F:\Cornea\ORTrack'
    settings.result_plot_path = 'F:\Cornea\ORTrack\test/result_plots'
    settings.results_path = 'F:\Cornea\ORTrack\test/tracking_results'    # Where to store tracking results
    settings.save_dir = 'F:\Cornea\ORTrack'
    settings.segmentation_path = 'F:\Cornea\ORTrack\test/segmentation_results'
    settings.tc128_path = 'F:\Cornea\data\TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'F:\Cornea\data\tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'F:\Cornea\data\trackingnet'
    settings.uav123_path = 'F:\Cornea\data\uav123'
    settings.uav_path = 'F:\Cornea\data\uav'
    settings.uavdt_path = 'F:\Cornea\data\uavdt'
    settings.visdrone2018_path = 'F:\Cornea\data\visdrone2018'
    settings.vot18_path = 'F:\Cornea\data\vot2018'
    settings.vot22_path = 'F:\Cornea\data\vot2022'
    settings.vot_path = 'F:\Cornea\data\VOT2019'
    settings.youtubevos_dir = ''

    return settings

