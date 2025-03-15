from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_Physio, Dataset_PEMS, Dataset_Epilepsy
import torch


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Exchange': Dataset_Custom,
    'Weather': Dataset_Custom,
    'ECL': Dataset_Custom,
    'ILI': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'HAR': Dataset_Physio,
    'EEG': Dataset_Physio,
    'PEMS03': Dataset_PEMS,
    'PEMS04': Dataset_PEMS,
    'PEMS07': Dataset_PEMS,
    'PEMS08': Dataset_PEMS,
    'Epilepsy': Dataset_Epilepsy,
}
def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.input_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        seasonal_patterns=args.seasonal_patterns
    )
    return data_set


