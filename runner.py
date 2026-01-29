from tqdm import tqdm
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.size'] = 14
from utils import *
from simulation import run_monotone, run_extragradient, run_admm, run_polyaksgm, run_mse_gd
from time import time

start_time = time()

nu = nx*2 # number of detector cells
save_used = True

ns_list = [984, 492, 246, 164, 123, 82, 41, 24, 12, 8]

use_total_budgets = False
total_budgets = np.array([1e5, 1e4, 1e3, 1e2])*984
total_budgets = np.array([1e7])*984
intensity_lists = []
for total_budget in total_budgets:
    intensity_list = [total_budget/ns for ns in ns_list]
    intensity_lists.append(intensity_list)
    use_total_budgets = True

total_views = 984

print(intensity_list)
print('ns_list', ns_list)
seed_list = [0, 737, 747, 777, 787, 320, 350, 330, 380]

plot_curves = True # only plot for seed=0 or debug seed=10
use_TV = True
use_noise = True
use_extragradient = False#True #algorithm
use_monotone = True #algorithm
use_mse_gd = False #algorithm
use_polyaksgm = False #algorithm
use_admm = False #algorithm
plot_quantiles = True  # If True, show median and shade between 25th and 75th percentiles. If False, show mean and shade +- stddev
foldername = f'multi_material_res_{nx}_lowstepsize'
foldernames_ns = [os.path.join(foldername, f'ns{ns}') for ns in ns_list]
foldernames_intensity = [os.path.join(foldername, f'intensity{intensity}') for intensity in intensity_list]

def run_experiment(ns, total_intensity, seed, expname, plot_curves):
    if seed != 0 and seed != 10:  # This is just to save time, since plots for different seeds are very similar
        plot_curves = False
    intensity = total_intensity
    maxtv = 0
    if use_TV:
        maxtv = oracle_tv
    
    foldername = os.path.join(expname, f'seed{seed}')
    os.makedirs(foldername, exist_ok=True)

    nl = ns*nu  # number of rays
    load_or_create_P(ns, nu, nl, foldername, use_saved = save_used)
    print('generating s')
    S = load_or_create_S(total_intensity, nl, foldername, use_saved = save_used)
    print('generating c')
    c = load_or_create_c(x_true, nl, S, seed, use_noise, foldername, use_saved = save_used)

    # Monotone
    x_store_iters_monotone = []
    runtimes_monotone = []
    if use_monotone:
        print("using monotone")
        try:
            save_name = f'intensity{intensity:1.2e}_ns{ns}_monotone_{stepsizes[0]}_TV{use_TV}_noise{use_noise}.npy'
        except: 
            save_name = f'{total_budget:1.2e}_ns{ns}_monotone_{stepsizes[0]}_TV{use_TV}_noise{use_noise}.npy'
            intensity = total_budget
            print('Different intensities detected in one experiment, the total intensity is ',total_intensity)
        if os.path.exists(os.path.join(foldername, f'iters_{save_name}')):
            print(f'reusing stored results for monotone operator')
            x_store_iters_monotone = np.load(os.path.join(foldername, f'iters_{save_name}'))
            runtimes_monotone = np.load(os.path.join(foldername, f'runtimes_{save_name}'))
        else:
            runningavgs, runtimes_monotone = run_monotone(c=c, nl=nl, S=S, maxtv=maxtv, total_intensity=total_intensity)
            x_store_iters_monotone = np.array([runningavg.iterates for runningavg in runningavgs])
            print(os.path.join(foldername, f'iters_monotone_{save_name}'))
            np.save(os.path.join(foldername, f'iters_monotone_{save_name}'), x_store_iters_monotone[0,-1,...][None,None,...])
            np.save(os.path.join(foldername, f'runtimes_monotone_{save_name}'), runtimes_monotone)#[0,-1,...][None,None,...])
        
    # Extragradient
    x_store_iters_extragradient = []
    runtimes_extragradient = []
    if use_extragradient:
        print("using extragradient")
        save_name = f'intensity{intensity:1.2e}_ns{ns}_extra_{stepsizes[0]}_TV{use_TV}_noise{use_noise}.npy'
        if os.path.exists(os.path.join(foldername, f'iters_{save_name}')):
            print(f'reusing stored results for extragradient')
            x_store_iters_extragradient = np.load(os.path.join(foldername, f'iters_{save_name}'))
            runtimes_extragradient = np.load(os.path.join(foldername, f'runtimes_{save_name}'))
        else:
            runningavgs, runtimes_extragradient = run_extragradient(c=c, nl=nl, S=S, maxtv=maxtv, total_intensity=total_intensity)
            x_store_iters_extragradient = np.array([runningavg.iterates for runningavg in runningavgs])
            np.save(os.path.join(foldername, f'iters_{save_name}'), x_store_iters_extragradient)
            np.save(os.path.join(foldername, f'runtimes_{save_name}'), runtimes_extragradient)
    # Polyak SGM
    x_store_iters_polyaksgm = []
    runtimes_polyaksgm = []
    if use_polyaksgm:
        print("using polyaksgm")
        save_name = f'intensity{intensity:0.0e}_ns{ns}_polyaksgm_{polyakstepsizes[0]}_TV{use_TV}_noise{use_noise}.npy'
        if os.path.exists(os.path.join(foldername, f'iters_')):
            print(f'reusing stored results for Polyak SGM')
            x_store_iters_polyaksgm = np.load(os.path.join(foldername, f'iters_{save_name}'))
            runtimes_polyaksgm = np.load(os.path.join(foldername, f'runtimes_{save_name}'))
        else:
            # x_store_iters_polyaksgm, runtimes_polyaksgm = run_polyaksgm_noopt(c=c, P=P, nl=nl, S=S, Tinner=1000)
            runningavgs, runtimes_polyaksgm = run_polyaksgm(c=c, P=P, nl=nl, S=S, xstar=x_true, maxtv=maxtv)
            x_store_iters_polyaksgm = np.array([runningavg.iterates for runningavg in runningavgs])
            np.save(os.path.join(foldername, f'iters_{save_name}'), x_store_iters_polyaksgm)
            np.save(os.path.join(foldername, f'runtimes_{save_name}'), runtimes_polyaksgm)
    # MSE Gradient Descent
    x_store_iters_msegd = []
    runtimes_msegd = []
    if use_mse_gd:
        save_name = f'intensity{intensity:0.0e}_ns{ns}_msegd_{gdstepsizes[0]}_TV{use_TV}_noise{use_noise}.npy'
        if os.path.exists(os.path.join(foldername, f'iters_{save_name}')):
            print(f'reusing stored results for MSE GD')
            x_store_iters_msegd = np.load(os.path.join(foldername, f'iters_{save_name}'))
            runtimes_msegd = np.load(os.path.join(foldername, f'runtimes_{save_name}'))
        else:
            runningavgs, runtimes_msegd = run_mse_gd(c=c, nl=nl, S=S, maxtv=maxtv)
            x_store_iters_msegd = np.array([runningavg.iterates for runningavg in runningavgs])
            np.save(os.path.join(foldername, f'iters_{save_name}'), x_store_iters_msegd)
            np.save(os.path.join(foldername, f'runtimes_{save_name}'), runtimes_msegd)
    # ADMM
    x_store_iters_admm = []
    runtimes_admm = []
    if use_admm:
        print("using ADMM")
        save_name = f'intensity{intensity:0.0e}_ns{ns}_admm_{sig_grid[0]}_TV{use_TV}_noise{use_noise}.npy'
        if os.path.exists(os.path.join(foldername, f'iters_{save_name}')):
            print(f'reusing stored results for ADMM')
            x_store_iters_admm = np.load(os.path.join(foldername, f'iters_{save_name}'))
            runtimes_admm = np.load(os.path.join(foldername, f'runtimes_{save_name}'))
        else:
            runningavgs, runtimes_admm = run_admm(c=c, P=P, nl=nl, S=S, maxtv=maxtv)
            x_store_iters_admm = np.array([runningavg.iterates for runningavg in runningavgs])
            np.save(os.path.join(foldername, f'iters_{save_name}'), x_store_iters_admm)
            np.save(os.path.join(foldername, f'runtimes_{save_name}'), runtimes_admm)

    print(f'***** using {ns} projections and {total_intensity} photons, with seed {seed} *****')
    if use_monotone:
        print(f'runtime for monotone was {np.mean(runtimes_monotone)} seconds')
    if use_extragradient:
        print(f'runtime for extragradient was {np.mean(runtimes_extragradient)} seconds')
    if use_mse_gd:
        print(f'runtime for MSE GD was {np.mean(runtimes_msegd)} seconds')
    if use_admm:
        print(f'runtime for admm was {np.mean(runtimes_admm)} seconds')
    if use_polyaksgm:
        print(f'runtime for polyakSGM was {np.mean(runtimes_polyaksgm)} seconds')

    # plot results: true & estimated images
    x_list = [x_true]

    title_list = ['Ground truth']
    filenames = [os.path.join(foldername, 'target.jpg')]
    # ADMM
    if use_admm:
        for isig in range(nsig):
            x_list.append(x_store_iters_admm[isig][-1])
            title_list.append('Estimated image (# iterations='+str(len(x_store_iters_admm[isig])-1)+\
                    ', ADMM parameter '+r'$\sigma=$'+str(sig_grid[isig])+')')
            filenames.append(os.path.join(foldername, f'recon_intensity{intensity}_ns{ns}_sigma{sig_grid[isig]}_TV{use_TV}_noise{use_noise}.jpg'))
    # Monotone
    if use_monotone:
        for istepsize in range(nstepsize):
            if istepsize == 0:
                x_bar[..., -1] = x_store_iters_monotone[istepsize][-1][...,-1]
                # print(x_store_iters_monotone[istepsize][-1].shape)
                # x_list.append(x_store_iters_monotone[istepsize][-1])
                x_list.append(x_bar)
                title_list.append('Estimated image (# iterations='+str(len(x_store_iters_monotone[istepsize])-1)+\
                        ', stepsize '+r'$\eta=$'+str(stepsizes[istepsize])+')')
                filenames.append(os.path.join(foldername, f'recon_intensity{intensity}_ns{ns}_stepsize{stepsizes[istepsize]}_TV{use_TV}_noise{use_noise}.jpg'))
    # Extragradient
    if use_extragradient:
        for istepsize in range(nstepsize):
            if istepsize == 0:
                x_bar[..., -1] = x_store_iters_extragradient[istepsize][-1][...,-1]
                x_list.append(x_bar)
                # x_list.append(x_store_iters_extragradient[istepsize][-1])
                title_list.append('Estimated image (# iterations='+str(len(x_store_iters_extragradient[istepsize])-1)+\
                        ', stepsize '+r'$\eta=$'+str(stepsizes[istepsize])+')')
                filenames.append(os.path.join(foldername, f'recon_intensity{intensity}_ns{ns}_extra_stepsize{stepsizes[istepsize]}_TV{use_TV}_noise{use_noise}.jpg'))
    # MSE Gradient Descent
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            if istepsize == 0:
                x_list.append(x_store_iters_msegd[istepsize][-1])
                title_list.append('Estimated image (# iterations='+str(len(x_store_iters_msegd[istepsize])-1)+\
                        ', stepsize '+r'$\eta=$'+str(gdstepsizes[istepsize])+')')
                filenames.append(os.path.join(foldername, f'recon_intensity{intensity}_ns{ns}_msegd_stepsize{gdstepsizes[istepsize]}_TV{use_TV}_noise{use_noise}.jpg'))
    # PolyakSGM
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            if istepsize == 0:
                x_list.append(x_store_iters_polyaksgm[istepsize][-1])
                title_list.append('Estimated image (# iterations='+str(len(x_store_iters_polyaksgm[istepsize])-1)+\
                        ', stepsize '+r'$\eta=$'+str(polyakstepsizes[istepsize])+')')
                filenames.append(os.path.join(foldername, f'recon_intensity{intensity}_ns{ns}_polyak{polyakstepsizes[istepsize]}_TV{use_TV}_noise{use_noise}.jpg'))


    plot_x(x_list, title_list, filenames)
    # np.save(os.path.join(foldername, 'recon_save.npy'),x_list[1])
    if not plot_curves:
        return x_store_iters_admm, x_store_iters_monotone, x_store_iters_polyaksgm, x_store_iters_extragradient, x_store_iters_msegd, runtimes_admm, runtimes_monotone, runtimes_polyaksgm, runtimes_extragradient, runtimes_msegd

    if use_admm:
        loss_by_iter_admm = []
        rmse_by_iter_admm = []
        norm_by_iter_admm = []
        avgloss_by_iter_admm = []
        avgrmse_by_iter_admm = []
        avgnorm_by_iter_admm = []
        iters_admm = []
    if use_monotone:
        loss_by_iter_monotone = []
        rmse_by_iter_monotone = []
        norm_by_iter_monotone = []
        avgloss_by_iter_monotone = []
        avgrmse_by_iter_monotone = []
        avgnorm_by_iter_monotone = []
        iters_monotone = []
    if use_extragradient:
        loss_by_iter_extragradient = []
        rmse_by_iter_extragradient = []
        norm_by_iter_extragradient = []
        avgloss_by_iter_extragradient = []
        avgrmse_by_iter_extragradient = []
        avgnorm_by_iter_extragradient = []
        iters_extragradient = []
    if use_mse_gd:
        loss_by_iter_msegd = []
        rmse_by_iter_msegd = []
        norm_by_iter_msegd = []
        avgloss_by_iter_msegd = []
        avgrmse_by_iter_msegd = []
        avgnorm_by_iter_msegd = []
        iters_msegd = []
    if use_polyaksgm:
        loss_by_iter_polyaksgm = []
        rmse_by_iter_polyaksgm = []
        norm_by_iter_polyaksgm = []
        avgloss_by_iter_polyaksgm = []
        avgrmse_by_iter_polyaksgm = []
        avgnorm_by_iter_polyaksgm = []
        iters_polyaksgm = []

    # ADMM
    if use_admm:
        for isig in range(nsig):
            iters_sig, loss_sig, rmse_sig, normratio_sig, avgloss_sig, avgrmse_sig = get_plot_curves(x_store_iters_admm[isig], c, S, P, nl, x_true, method='admm')
            loss_by_iter_admm.append(loss_sig)
            rmse_by_iter_admm.append(rmse_sig)
            norm_by_iter_admm.append(normratio_sig)
            avgloss_by_iter_admm.append(avgloss_sig)
            avgrmse_by_iter_admm.append(avgrmse_sig)
            iters_admm.append(iters_sig)
        loss_by_iter_admm = np.array(loss_by_iter_admm)
        rmse_by_iter_admm = np.array(rmse_by_iter_admm)
        norm_by_iter_admm = np.array(norm_by_iter_admm)
        avgloss_by_iter_admm = np.array(avgloss_by_iter_admm)
        avgrmse_by_iter_admm = np.array(avgrmse_by_iter_admm)
        avgnorm_by_iter_admm = np.array(avgnorm_by_iter_admm)
        iters_admm = np.array(iters_admm)
    # Monotone
    if use_monotone:
        for istepsize in range(nstepsize):
            iters_stepsize, loss_stepsize, rmse_stepsize, normratio_stepsize, avgloss_stepsize, avgrmse_stepsize = get_plot_curves(x_store_iters_monotone[istepsize], c,  S, nl, x_true, method='monotone')
            loss_by_iter_monotone.append(loss_stepsize)
            rmse_by_iter_monotone.append(rmse_stepsize)
            norm_by_iter_monotone.append(normratio_stepsize)
            avgloss_by_iter_monotone.append(avgloss_stepsize)
            avgrmse_by_iter_monotone.append(avgrmse_stepsize)
            iters_monotone.append(iters_stepsize)
        loss_by_iter_monotone = np.array(loss_by_iter_monotone)
        rmse_by_iter_monotone = np.array(rmse_by_iter_monotone)
        norm_by_iter_monotone = np.array(norm_by_iter_monotone)
        avgloss_by_iter_monotone = np.array(avgloss_by_iter_monotone)
        avgrmse_by_iter_monotone = np.array(avgrmse_by_iter_monotone)
        avgnorm_by_iter_monotone = np.array(avgnorm_by_iter_monotone)
        iters_monotone = np.array(iters_monotone)
    # Extragradient
    if use_extragradient:
        for istepsize in range(nstepsize):
            iters_stepsize, loss_stepsize, rmse_stepsize, normratio_stepsize, avgloss_stepsize, avgrmse_stepsize = get_plot_curves(x_store_iters_extragradient[istepsize], c, S, nl, x_true, method='extragradient')
            loss_by_iter_extragradient.append(loss_stepsize)
            rmse_by_iter_extragradient.append(rmse_stepsize)
            norm_by_iter_extragradient.append(normratio_stepsize)
            avgloss_by_iter_extragradient.append(avgloss_stepsize)
            avgrmse_by_iter_extragradient.append(avgrmse_stepsize)
            iters_extragradient.append(iters_stepsize)
        loss_by_iter_extragradient = np.array(loss_by_iter_extragradient)
        rmse_by_iter_extragradient = np.array(rmse_by_iter_extragradient)
        norm_by_iter_extragradient = np.array(norm_by_iter_extragradient)
        avgloss_by_iter_extragradient = np.array(avgloss_by_iter_extragradient)
        avgrmse_by_iter_extragradient = np.array(avgrmse_by_iter_extragradient)
        avgnorm_by_iter_extragradient = np.array(avgnorm_by_iter_extragradient)
        iters_extragradient = np.array(iters_extragradient)
    # MSE Gradient Descent
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            iters_stepsize, loss_stepsize, rmse_stepsize, normratio_stepsize, avgloss_stepsize, avgrmse_stepsize = get_plot_curves(x_store_iters_msegd[istepsize], c, S, P, nl, x_true, method='mse_gd')
            loss_by_iter_msegd.append(loss_stepsize)
            rmse_by_iter_msegd.append(rmse_stepsize)
            norm_by_iter_msegd.append(normratio_stepsize)
            avgloss_by_iter_msegd.append(avgloss_stepsize)
            avgrmse_by_iter_msegd.append(avgrmse_stepsize)
            iters_msegd.append(iters_stepsize)
        loss_by_iter_msegd = np.array(loss_by_iter_msegd)
        rmse_by_iter_msegd = np.array(rmse_by_iter_msegd)
        norm_by_iter_msegd = np.array(norm_by_iter_msegd)
        avgloss_by_iter_msegd = np.array(avgloss_by_iter_msegd)
        avgrmse_by_iter_msegd = np.array(avgrmse_by_iter_msegd)
        avgnorm_by_iter_msegd = np.array(avgnorm_by_iter_msegd)
        iters_msegd = np.array(iters_msegd)
    # PolyakSGM
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            iters_stepsize, loss_stepsize, rmse_stepsize, normratio_stepsize, avgloss_stepsize, avgrmse_stepsize = get_plot_curves(x_store_iters_polyaksgm[istepsize], c, S, P, nl, x_true, method='polyaksgm')
            loss_by_iter_polyaksgm.append(loss_stepsize)
            rmse_by_iter_polyaksgm.append(rmse_stepsize)
            norm_by_iter_polyaksgm.append(normratio_stepsize)
            avgloss_by_iter_polyaksgm.append(avgloss_stepsize)
            avgrmse_by_iter_polyaksgm.append(avgrmse_stepsize)
            iters_polyaksgm.append(iters_stepsize)
        loss_by_iter_polyaksgm = np.array(loss_by_iter_polyaksgm)
        rmse_by_iter_polyaksgm = np.array(rmse_by_iter_polyaksgm)
        norm_by_iter_polyaksgm = np.array(norm_by_iter_polyaksgm)
        avgloss_by_iter_polyaksgm = np.array(avgloss_by_iter_polyaksgm)
        avgrmse_by_iter_polyaksgm = np.array(avgrmse_by_iter_polyaksgm)
        avgnorm_by_iter_polyaksgm = np.array(avgnorm_by_iter_polyaksgm)
        iters_polyaksgm = np.array(iters_polyaksgm)


    plt.figure()
    if use_admm:
        for isig in range(nsig):
            plt.plot(1+iters_admm[isig],loss_by_iter_admm[isig],label=\
                    r'ADMM $\sigma=$'+str(sig_grid[isig]), c=color_dict['admm'], marker=shape_dict['admm'])
    if use_monotone:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_monotone[istepsize],loss_by_iter_monotone[istepsize],label=\
                    r'Monotone $\eta=$'+str(stepsizes[istepsize]), c=color_dict['monotone'], marker=shape_dict['monotone'])
    if use_extragradient:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_extragradient[istepsize],loss_by_iter_extragradient[istepsize],label=\
                    r'Extragradient $\eta=$'+str(stepsizes[istepsize]), c=color_dict['extragradient'], marker=shape_dict['extragradient'])
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            plt.plot(1+iters_msegd[istepsize],loss_by_iter_msegd[istepsize],label=\
                    r'MSE GD $\eta=$'+str(gdstepsizes[istepsize]), c=color_dict['msegd'], marker=shape_dict['msegd'])
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            plt.plot(1+iters_polyaksgm[istepsize],loss_by_iter_polyaksgm[istepsize],label=\
                    r'PolyakSGM $\eta=$'+str(polyakstepsizes[istepsize]), c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel('Loss'+r'$(Px_t)$')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(foldername, f'algspecific_loss_intensity{intensity:0.0e}_ns{ns}_TV{use_TV}_noise{use_noise}.jpg'))
    plt.close()

    plt.figure()
    if use_admm:
        for isig in range(nsig):
            plt.plot(1+iters_admm[isig],avgloss_by_iter_admm[isig],label=\
                    r'ADMM $\sigma=$'+str(sig_grid[isig]), c=color_dict['admm'], marker=shape_dict['admm'])
    if use_monotone:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_monotone[istepsize],avgloss_by_iter_monotone[istepsize],label=\
                    r'Monotone $\eta=$'+str(stepsizes[istepsize]), c=color_dict['monotone'], marker=shape_dict['monotone'])
    if use_extragradient:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_extragradient[istepsize],avgloss_by_iter_extragradient[istepsize],label=\
                    r'Extragradient $\eta=$'+str(stepsizes[istepsize]), c=color_dict['extragradient'], marker=shape_dict['extragradient'])
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            plt.plot(1+iters_msegd[istepsize],avgloss_by_iter_msegd[istepsize],label=\
                    r'MSE GD $\eta=$'+str(gdstepsizes[istepsize]), c=color_dict['msegd'], marker=shape_dict['msegd'])
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            plt.plot(1+iters_polyaksgm[istepsize],avgloss_by_iter_polyaksgm[istepsize],label=\
                    r'PolyakSGM $\eta=$'+str(polyakstepsizes[istepsize]), c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel('Loss'+r'$(P\bar{x}_t)$')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(foldername, f'algspecific_avg_loss_intensity{intensity:0.0e}_ns{ns}_TV{use_TV}_noise{use_noise}.jpg'))
    plt.close()


    plt.figure()
    if use_admm:
        for isig in range(nsig):
            plt.plot(1+iters_admm[isig],rmse_by_iter_admm[isig],label=\
                    r'ADMM $\sigma=$'+str(sig_grid[isig]), c=color_dict['admm'], marker=shape_dict['admm'])
    if use_monotone:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_monotone[istepsize],rmse_by_iter_monotone[istepsize],label=\
                    r'Monotone $\eta=$'+str(stepsizes[istepsize]), c=color_dict['monotone'], marker=shape_dict['monotone'])
    if use_extragradient:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_extragradient[istepsize],rmse_by_iter_extragradient[istepsize],label=\
                    r'Extragradient $\eta=$'+str(stepsizes[istepsize]), c=color_dict['extragradient'], marker=shape_dict['extragradient'])
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            plt.plot(1+iters_msegd[istepsize],rmse_by_iter_msegd[istepsize],label=\
                    r'MSE GD $\eta=$'+str(gdstepsizes[istepsize]), c=color_dict['msegd'], marker=shape_dict['msegd'])
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            plt.plot(1+iters_polyaksgm[istepsize],rmse_by_iter_polyaksgm[istepsize],label=\
                    r'PolyakSGM $\eta=$'+str(polyakstepsizes[istepsize]), c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel('RMSE'+r'$(x_t)$')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(foldername, f'image_rmse_intensity{intensity:0.0e}_ns{ns}_TV{use_TV}_noise{use_noise}.jpg'))
    plt.close()

    plt.figure()
    if use_admm:
        for isig in range(nsig):
            plt.plot(1+iters_admm[isig],avgrmse_by_iter_admm[isig],label=\
                    r'ADMM $\sigma=$'+str(sig_grid[isig]), c=color_dict['admm'], marker=shape_dict['admm'])
    if use_monotone:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_monotone[istepsize],avgrmse_by_iter_monotone[istepsize],label=\
                    r'Monotone $\eta=$'+str(stepsizes[istepsize]), c=color_dict['monotone'], marker=shape_dict['monotone'])
    if use_extragradient:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_extragradient[istepsize],avgrmse_by_iter_extragradient[istepsize],label=\
                    r'Extragradient $\eta=$'+str(stepsizes[istepsize]), c=color_dict['extragradient'], marker=shape_dict['extragradient'])
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            plt.plot(1+iters_msegd[istepsize],avgrmse_by_iter_msegd[istepsize],label=\
                    r'MSE GD $\eta=$'+str(gdstepsizes[istepsize]), c=color_dict['msegd'], marker=shape_dict['msegd'])
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            plt.plot(1+iters_polyaksgm[istepsize],avgrmse_by_iter_polyaksgm[istepsize],label=\
                    r'PolyakSGM $\eta=$'+str(polyakstepsizes[istepsize]), c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel('RMSE'+r'$(\bar{x}_t)$')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(foldername, f'image_avg_rmse_intensity{intensity:0.0e}_ns{ns}_TV{use_TV}_noise{use_noise}.jpg'))
    plt.close()


    plt.figure()
    if use_admm:
        for isig in range(nsig):
            plt.plot(1+iters_admm[isig],norm_by_iter_admm[isig],label=\
                    r'ADMM $\sigma=$'+str(sig_grid[isig]), c=color_dict['admm'], marker=shape_dict['admm'])
    if use_monotone:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_monotone[istepsize],norm_by_iter_monotone[istepsize],label=\
                    r'Monotone $\eta=$'+str(stepsizes[istepsize]), c=color_dict['monotone'], marker=shape_dict['monotone'])
    if use_extragradient:
        for istepsize in range(nstepsize):
            plt.plot(1+iters_extragradient[istepsize],norm_by_iter_extragradient[istepsize],label=\
                    r'Exragradient $\eta=$'+str(stepsizes[istepsize]), c=color_dict['extragradient'], marker=shape_dict['extragradient'])
    if use_mse_gd:
        for istepsize in range(ngdstepsize):
            plt.plot(1+iters_msegd[istepsize],norm_by_iter_msegd[istepsize],label=\
                    r'MSE GD $\eta=$'+str(gdstepsizes[istepsize]), c=color_dict['msegd'], marker=shape_dict['msegd'])
    if use_polyaksgm:
        for istepsize in range(npolyakstepsize):
            plt.plot(1+iters_polyaksgm[istepsize],norm_by_iter_polyaksgm[istepsize],label=\
                    r'PolyakSGM $\eta=$'+str(polyakstepsizes[istepsize]), c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
    plt.xscale('log')
    plt.xlabel('Iteration t')
    plt.ylabel(r'$||\bar{x}_t - \bar{x}_{t-1}||$')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(foldername, f'image_intensity{intensity:0.0e}_ns{ns}_norm_TV{use_TV}_noise{use_noise}.jpg'))
    plt.close()

    return x_store_iters_admm, x_store_iters_monotone, x_store_iters_polyaksgm, x_store_iters_extragradient, x_store_iters_msegd, runtimes_admm, runtimes_monotone, runtimes_polyaksgm, runtimes_extragradient, runtimes_msegd


def plot_ranges(data):
    if plot_quantiles:
        results = np.quantile(data, q=[0.25, 0.5, 0.75], axis=-1)
        middle = results[1]
        low = results[0]
        high = results[2]
    else:
        middle = np.mean(data, axis=-1)
        std = np.std(data, axis=-1)
        low = middle - std
        high = middle + std
    return low, middle, high


if len(ns_list) > 0 and len(intensity_list) > 0:
    if use_admm:
        rmse_admm = np.zeros((len(intensity_list) ,len(ns_list), nsig, len(seed_list)))
        avgrmse_admm, runtimes_admm = np.zeros_like(rmse_admm), np.zeros_like(rmse_admm)
    if use_polyaksgm:
        rmse_polyaksgm = np.zeros((len(intensity_list), len(ns_list), npolyakstepsize, len(seed_list)))
        avgrmse_polyaksgm, runtimes_polyaksgm = np.zeros_like(rmse_polyaksgm), np.zeros_like(rmse_polyaksgm)
    if use_monotone:
        rmse_monotone = np.zeros((len(intensity_list), len(ns_list), nstepsize, len(seed_list)))
        avgrmse_monotone, runtimes_monotone = np.zeros_like(rmse_monotone), np.zeros_like(rmse_monotone)
    if use_extragradient:
        rmse_extragradient = np.zeros((len(intensity_list), len(ns_list), nstepsize, len(seed_list)))
        avgrmse_extragradient, runtimes_extragradient = np.zeros_like(rmse_extragradient), np.zeros_like(rmse_extragradient)
    if use_mse_gd:
        rmse_msegd = np.zeros((len(intensity_list), len(ns_list), ngdstepsize, len(seed_list)))
        avgrmse_msegd, runtimes_msegd = np.zeros_like(rmse_msegd), np.zeros_like(rmse_msegd)
    for intensities in intensity_lists:
        print(intensities)
        for idx_intens, intensity in tqdm(enumerate(intensities), desc = 'intensity'):
            ns = ns_list[idx_intens]
            print('now optimizing with intensity:',intensity,'and ns:', ns)
            idx = 0
            fname = os.path.join(foldername, f'intensity{intensity}_ns{ns}')
            for iseed in tqdm(range(len(seed_list))):
                x_store_iters_admm, x_store_iters_monotone, x_store_iters_polyaksgm, x_store_iters_extragradient, x_store_iters_msegd, runtime_admm, runtime_monotone, runtime_polyaksgm, runtime_extragradient, runtime_msegd = run_experiment(ns=ns, 
                                                                                                                                                                                                                                                total_intensity=intensity, 
                                                                                                                                                                                                                                                seed=seed_list[iseed], 
                                                                                                                                                                                                                                                expname=fname, 
                                                                                                                                                                                                                                                plot_curves=plot_curves)
                if use_admm:
                    for isig in range(nsig):
                        rmse_admm[idx_intens, idx,isig,iseed] = np.sqrt(np.mean((x_store_iters_admm[isig][-1]-x_true)**2))
                        avgiter = np.mean(x_store_iters_admm[isig][int(len(x_store_iters_admm[isig])/2):], axis=0)
                        avgrmse_admm[idx_intens, idx,isig,iseed] = np.sqrt(np.mean((avgiter-x_true)**2))
                        runtimes_admm[idx_intens, idx,isig,iseed] = runtime_admm[0]  # Note this won't correctly handle multiple sigmas
                if use_monotone:
                    for istepsize in range(nstepsize):
                        rmse_monotone[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((x_store_iters_monotone[istepsize][-1]-x_true)**2))
                        avgiter = np.mean(x_store_iters_monotone[istepsize][int(len(x_store_iters_monotone[istepsize])/2):], axis=0)
                        avgrmse_monotone[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((avgiter-x_true)**2))
                        runtimes_monotone[idx_intens, idx,istepsize,iseed] = runtime_monotone[0]  # Note this won't correctly handle multiple stepsizes
                if use_extragradient:
                    for istepsize in range(nstepsize):
                        rmse_extragradient[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((x_store_iters_extragradient[istepsize][-1]-x_true)**2))
                        avgiter = np.mean(x_store_iters_extragradient[istepsize][int(len(x_store_iters_extragradient[istepsize])/2):], axis=0)
                        avgrmse_extragradient[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((avgiter-x_true)**2))
                        runtimes_extragradient[idx_intens, idx,istepsize,iseed] = runtime_extragradient[0]  # Note this won't correctly handle multiple stepsizes
                if use_mse_gd:
                    for istepsize in range(ngdstepsize):
                        rmse_msegd[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((x_store_iters_msegd[istepsize][-1]-x_true)**2))
                        avgiter = np.mean(x_store_iters_msegd[istepsize][int(len(x_store_iters_msegd[istepsize])/2):], axis=0)
                        avgrmse_msegd[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((avgiter-x_true)**2))
                        runtimes_msegd[idx_intens, idx,istepsize,iseed] = runtime_msegd[0]  # Note this won't correctly handle multiple stepsizes
                if use_polyaksgm:
                    for istepsize in range(npolyakstepsize):
                        rmse_polyaksgm[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((x_store_iters_polyaksgm[istepsize][-1]-x_true)**2))
                        avgiter = np.mean(x_store_iters_polyaksgm[istepsize][int(len(x_store_iters_polyaksgm[istepsize])/2):], axis=0)
                        avgrmse_polyaksgm[idx_intens, idx,istepsize,iseed] = np.sqrt(np.mean((avgiter-x_true)**2))
                        runtimes_polyaksgm[idx_intens, idx,istepsize,iseed] = runtime_polyaksgm[0]  # Note this won't correctly handle multiple stepsizes

            plt.figure()
            if use_admm:
                for isig in range(nsig):
                    low, middle, high = plot_ranges(rmse_admm[idx_intens,:,isig,:])
                    plt.plot(ns_list, middle, label=r'ADMM', c=color_dict['admm'], marker=shape_dict['admm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['admm'], alpha=0.5)
            if use_monotone:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(rmse_monotone[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Monotone', c=color_dict['monotone'], marker=shape_dict['monotone'])
                    plt.fill_between(ns_list, low, high, color=color_dict['monotone'], alpha=0.5)
            if use_extragradient:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(rmse_extragradient[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Extragradient', c=color_dict['extragradient'], marker=shape_dict['extragradient'])
                    plt.fill_between(ns_list, low, high, color=color_dict['extragradient'], alpha=0.5)
            if use_mse_gd:
                for istepsize in range(ngdstepsize):
                    low, middle, high = plot_ranges(rmse_msegd[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'MSE GD', c=color_dict['msegd'], marker=shape_dict['msegd'])
                    plt.fill_between(ns_list, low, high, color=color_dict['msegd'], alpha=0.5)
            if use_polyaksgm:
                for istepsize in range(npolyakstepsize):
                    low, middle, high = plot_ranges(rmse_polyaksgm[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'PolyakSGM', c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['polyaksgm'], alpha=0.5)
            plt.yscale('log')
            plt.xlabel('Number of Projections')
            plt.ylabel('RMSE'+r'$(x_t)$')
            plt.legend(loc = 'upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(foldername, f'RMSE_intensity{intensity}_ns_TV{use_TV}_noise{use_noise}.jpg'))
            plt.close()
            
            plt.figure()
            if use_admm:
                for isig in range(nsig):
                    low, middle, high = plot_ranges(avgrmse_admm[idx_intens,:, isig, :])
                    plt.plot(ns_list, middle, label=r'ADMM', c=color_dict['admm'], marker=shape_dict['admm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['admm'], alpha=0.5)
            if use_monotone:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(avgrmse_monotone[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Monotone', c=color_dict['monotone'], marker=shape_dict['monotone'])
                    plt.fill_between(ns_list, low, high, color=color_dict['monotone'], alpha=0.5)
            if use_extragradient:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(avgrmse_extragradient[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Extragradient', c=color_dict['extragradient'], marker=shape_dict['extragradient'])
                    plt.fill_between(ns_list, low, high, color=color_dict['extragradient'], alpha=0.5)
            if use_mse_gd:
                for istepsize in range(ngdstepsize):
                    low, middle, high = plot_ranges(avgrmse_msegd[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'MSE GD', c=color_dict['msegd'], marker=shape_dict['msegd'])
                    plt.fill_between(ns_list, low, high, color=color_dict['msegd'], alpha=0.5)
            if use_polyaksgm:    
                for istepsize in range(npolyakstepsize):
                    low, middle, high = plot_ranges(avgrmse_polyaksgm[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'PolyakSGM', c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['polyaksgm'], alpha=0.5)
            plt.yscale('log')
            plt.xlabel('Number of Projections')
            plt.ylabel('RMSE'+r'$(\bar{x}_t)$')
            plt.legend(loc = 'upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(foldername, f'RMSE_avg_intensity{intensity}_ns_TV{use_TV}_noise{use_noise}.jpg'))
            plt.close()

            plt.figure()
            if use_admm:
                for isig in range(nsig):
                    low, middle, high = plot_ranges(runtimes_admm[idx_intens,:, isig, :])
                    plt.plot(ns_list, middle, label=r'ADMM', c=color_dict['admm'], marker=shape_dict['admm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['admm'], alpha=0.5)
            if use_monotone:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(runtimes_monotone[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Monotone', c=color_dict['monotone'], marker=shape_dict['monotone'])
                    plt.fill_between(ns_list, low, high, color=color_dict['monotone'], alpha=0.5)
            if use_extragradient:
                for istepsize in range(nstepsize):
                    low, middle, high = plot_ranges(runtimes_extragradient[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'Extragradient', c=color_dict['extragradient'], marker=shape_dict['extragradient'])
                    plt.fill_between(ns_list, low, high, color=color_dict['extragradient'], alpha=0.5)
            if use_mse_gd:
                for istepsize in range(ngdstepsize):
                    low, middle, high = plot_ranges(runtimes_msegd[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'MSE GD', c=color_dict['msegd'], marker=shape_dict['msegd'])
                    plt.fill_between(ns_list, low, high, color=color_dict['msegd'], alpha=0.5)
            if use_polyaksgm:
                for istepsize in range(npolyakstepsize):
                    low, middle, high = plot_ranges(runtimes_polyaksgm[idx_intens,:,istepsize,:])
                    plt.plot(ns_list, middle, label=r'PolyakSGM', c=color_dict['polyaksgm'], marker=shape_dict['polyaksgm'])
                    plt.fill_between(ns_list, low, high, color=color_dict['polyaksgm'], alpha=0.5)
            # plt.yscale('log')
            plt.xlabel('Number of Projections')
            plt.ylabel('Runtime (s)')
            plt.legend(loc = 'upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(foldername, f'Time_intensity{intensity}_ns_TV{use_TV}_noise{use_noise}.jpg'))
            plt.close()

total_seconds = time() - start_time

hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)

print(f'Total Time spent: {hours:02d}:{minutes:02d}:{seconds:02d} with step_size = {stepsizes}')
