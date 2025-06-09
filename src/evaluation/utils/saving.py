import torch


def save_best_stats_txt(save_path, best_n_spurious, best_metric, best_RMSE, best_DTWD, best_FD, gpu_status, i):
    f = open(save_path + 'best_model.txt', 'w')
    f.write('Number of unsuccessful trajectories: ' + str(best_n_spurious) + '\n')
    f.write('RMSE + DTWD + FD: ' + str(best_metric) + '\n')
    f.write('RMSE: ' + str(best_RMSE) + '\n')
    f.write('DTWD: ' + str(best_DTWD) + '\n')
    f.write('FD: ' + str(best_FD) + '\n')
    f.write('Iteration number: ' + str(i) + '\n')
    f.write('\n\n ###### GPU information ###### \n')
    f.write(gpu_status)
    f.close()


def save_stats_txt(save_path, best_n_spurious, best_metric, best_RMSE, best_DTWD, best_FD, i):
    f = open(save_path + 'training_evaluation_summary.txt', 'a')
    f.write('Iteration number: ' + str(i) + '\n')
    f.write('Number of unsuccessful trajectories: ' + str(best_n_spurious) + '\n')
    f.write('RMSE + DTWD + FD: ' + str(best_metric) + '\n')
    f.write('RMSE: ' + str(best_RMSE) + '\n')
    f.write('DTWD: ' + str(best_DTWD) + '\n')
    f.write('FD: ' + str(best_FD) + '\n\n')

    f.close()

def save_stats_txt_continuous(save_path: str,
                              chamfer: list,
                              oriented_latent_error: list,
                              spurious_percentage: list,
                              iteration: int):

    full_path = save_path.rstrip("/") + "/shape_metrics.txt"

    with open(full_path, "w") as f:
        f.write("best_stat:\n")
        f.write("iteration number: " + str(iteration) + '\n')

        f.write("chamfer distance:\n")
        f.write('[ ' + ", ".join(str(c) for c in chamfer) + ' ]\n')

        f.write("orientation error latent space:\n")
        f.write('[ ' + ", ".join(str(e) for e in oriented_latent_error) + ' ]\n')

        f.write("spurious %:\n")
        f.write('[ ' + ", ".join(str(s) for s in spurious_percentage) + ' ]\n')


def check_gpu():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_status = ''
    gpu_status += '\nUsing device: ' + str(device) + '\n\n'

    # Additional Info when using cuda
    if device.type == 'cuda':
        gpu_status += torch.cuda.get_device_name(0) + '\n'
        gpu_status += 'Memory Usage:\n'
        gpu_status += 'Allocated: ' + str(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)) + ' GB\n'
        gpu_status += 'Cached:    ' + str(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)) + ' GB\n'

    return gpu_status
