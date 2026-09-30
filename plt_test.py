from utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--ckptsR', default='ckpts',
                    help="Name of the ckpts_master_relu folder")
parser.add_argument('--ckptsT', default='ckpts',
                    help="Name of the ckpts_hgq_relu folder")
parser.add_argument('--ckptsL', default='ckpts',
                    help="Name of the ckpts_hgq_relu_loss folder")
parser.add_argument('--ckptsS', default='ckpts',
                    help="Name of the ckpts_hgq_recoil folder")
parser.add_argument('--ckptsSave', default='ckpts',
                    help="Name of the ckpts folder to save graphs")


args = parser.parse_args()
#a=load(args.ckpts + '/' +args.restore_file+ '.resolutions')
a_relu=load(args.ckptsR + '/best.resolutions')    # master relu response tune
a_tune=load(args.ckptsT + '/best.resolutions')    # hgq relu response tune
a_loss=load(args.ckptsL + '/best.resolutions')    # hgq relu loss function
a_sig=load(args.ckptsS + '/best.resolutions')    # hgq relu recoil loss function
colors_relu = {
#    'pfMET': 'black',
    'puppiMET': 'blue',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'purple',
}
colors_tune = {
#    'pfMET': 'black',
    'puppiMET': 'blue',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'red',
}
colors_loss = {
#    'pfMET': 'black',
    'puppiMET': 'blue',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'magenta',
}
colors_sig = {
#    'pfMET': 'black',
    'puppiMET': 'blue',
#    'deepMETResponse': 'blue',
#    'deepMETResolution': 'green',
    'MET':  'orange',
}

label_arr = {
    'MET':     'Graph MET' ,
#    'pfMET':    'PF MET',
    'puppiMET': 'PUPPI MET',
#    'deepMETResponse': 'DeepMETResponse',
#    'deepMETResolution': 'DeepMETResolution',
}
resolutions_arr = {
    'MET':      [[],[],[]],
#    'pfMET':    [[],[],[]],
    'puppiMET': [[],[],[]],
#    'deepMETResponse': [[],[],[]],
#    'deepMETResolution': [[],[],[]],
}
for key in resolutions_arr:
         plt.figure(1)
         if key == 'puppiMET':
            xx_r = a_relu[key]['u_perp_resolution'][1][0:20]
            yy_r = a_relu[key]['u_perp_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label='puppiMET')
         else:
            xx_r = a_relu[key]['u_perp_resolution'][1][0:20]
            yy_r = a_relu[key]['u_perp_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label=f'{label_arr[key]}_master_relu_response_tune')
            xx_t = a_tune[key]['u_perp_resolution'][1][0:20]
            yy_t = a_tune[key]['u_perp_resolution'][0]
            plt.plot(xx_t, yy_t, 'o-', color=colors_tune[key], label=f'{label_arr[key]}_hgq_relu_response_tune')
            xx_s = a_sig[key]['u_perp_resolution'][1][0:20]
            yy_s = a_sig[key]['u_perp_resolution'][0]
            plt.plot(xx_s, yy_s, 's-', color=colors_sig[key], label=f'{label_arr[key]}_hgq_relu_recoil_loss')
            xx_l = a_loss[key]['u_perp_resolution'][1][0:20]
            yy_l = a_loss[key]['u_perp_resolution'][0]
            plt.plot(xx_l, yy_l, '^-', color=colors_loss[key], label=f'{label_arr[key]}_hgq_relu_loss_function')

         plt.figure(2)
         if key == 'puppiMET':
            xx_r = a_relu[key]['u_perp_scaled_resolution'][1][0:20]
            yy_r = a_relu[key]['u_perp_scaled_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label='puppiMET')
         else:
            xx_r = a_relu[key]['u_perp_scaled_resolution'][1][0:20]
            yy_r = a_relu[key]['u_perp_scaled_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label=f'{label_arr[key]}_master_relu_response_tune')
            xx_t = a_tune[key]['u_perp_scaled_resolution'][1][0:20]
            yy_t = a_tune[key]['u_perp_scaled_resolution'][0]
            plt.plot(xx_t, yy_t, 'o-', color=colors_tune[key], label=f'{label_arr[key]}_hgq_relu_response_tune')
            xx_s = a_sig[key]['u_perp_scaled_resolution'][1][0:20]
            yy_s = a_sig[key]['u_perp_scaled_resolution'][0]
            plt.plot(xx_s, yy_s, 's-', color=colors_sig[key], label=f'{label_arr[key]}_hgq_relu_recoil_loss')
            xx_l = a_loss[key]['u_perp_scaled_resolution'][1][0:20]
            yy_l = a_loss[key]['u_perp_scaled_resolution'][0]
            plt.plot(xx_l, yy_l, '^-', color=colors_loss[key], label=f'{label_arr[key]}_hgq_relu_loss_function')

         plt.figure(3)
         if key == 'puppiMET':
            xx_r = a_relu[key]['u_par_resolution'][1][0:20]
            yy_r = a_relu[key]['u_par_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label='puppiMET')
         else:
            xx_r = a_relu[key]['u_par_resolution'][1][0:20]
            yy_r = a_relu[key]['u_par_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label=f'{label_arr[key]}_master_relu_response_tune')
            xx_t = a_tune[key]['u_par_resolution'][1][0:20]
            yy_t = a_tune[key]['u_par_resolution'][0]
            plt.plot(xx_t, yy_t, 'o-', color=colors_tune[key], label=f'{label_arr[key]}_hgq_relu_response_tune')
            xx_s = a_sig[key]['u_par_resolution'][1][0:20]
            yy_s = a_sig[key]['u_par_resolution'][0]
            plt.plot(xx_s, yy_s, 's-', color=colors_sig[key], label=f'{label_arr[key]}_hgq_relu_recoil_loss')
            xx_l = a_loss[key]['u_par_resolution'][1][0:20]
            yy_l = a_loss[key]['u_par_resolution'][0]
            plt.plot(xx_l, yy_l, '^-', color=colors_loss[key], label=f'{label_arr[key]}_hgq_relu_loss_function')

         plt.figure(4)
         if key == 'puppiMET':
            xx_r = a_relu[key]['u_par_scaled_resolution'][1][0:20]
            yy_r = a_relu[key]['u_par_scaled_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label='puppiMET')
         else:
            xx_r = a_relu[key]['u_par_scaled_resolution'][1][0:20]
            yy_r = a_relu[key]['u_par_scaled_resolution'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label=f'{label_arr[key]}_master_relu_response_tune')
            xx_t = a_tune[key]['u_par_scaled_resolution'][1][0:20]
            yy_t = a_tune[key]['u_par_scaled_resolution'][0]
            plt.plot(xx_t, yy_t, 'o-', color=colors_tune[key], label=f'{label_arr[key]}_hgq_relu_response_tune')
            xx_s = a_sig[key]['u_par_scaled_resolution'][1][0:20]
            yy_s = a_sig[key]['u_par_scaled_resolution'][0]
            plt.plot(xx_s, yy_s, 's-', color=colors_sig[key], label=f'{label_arr[key]}_hgq_relu_recoil_loss')
            xx_l = a_loss[key]['u_par_scaled_resolution'][1][0:20]
            yy_l = a_loss[key]['u_par_scaled_resolution'][0]
            plt.plot(xx_l, yy_l, '^-', color=colors_loss[key], label=f'{label_arr[key]}_hgq_relu_loss_function')
         
         plt.figure(5)
         if key == 'puppiMET':
            xx_r = a_relu[key]['R'][1][0:20]
            yy_r = a_relu[key]['R'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label='puppiMET')
         else:
            xx_r = a_relu[key]['R'][1][0:20]
            yy_r = a_relu[key]['R'][0]
            plt.plot(xx_r, yy_r, 'o-', color=colors_relu[key], label=f'{label_arr[key]}_master_relu_response_tune')
            xx_t = a_tune[key]['R'][1][0:20]
            yy_t = a_tune[key]['R'][0]
            plt.plot(xx_t, yy_t, 'o-', color=colors_tune[key], label=f'{label_arr[key]}_hgq_relu_response_tune')
            xx_s = a_sig[key]['R'][1][0:20]
            yy_s = a_sig[key]['R'][0]
            plt.plot(xx_s, yy_s, 's-', color=colors_sig[key], label=f'{label_arr[key]}_hgq_relu_recoil_loss')
            xx_l = a_loss[key]['R'][1][0:20]
            yy_l = a_loss[key]['R'][0]
            plt.plot(xx_l, yy_l, '^-', color=colors_loss[key], label=f'{label_arr[key]}_hgq_relu_loss_function')

if(True):
    model_dir=args.ckptsSave + '/'
    #model_dir=args.ckpts+'/'+args.restore_file+'_'
    plt.figure(1)
    plt.axis([0, 400, 0, 50])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp.png')
    plt.clf()
    plt.close()

    plt.figure(2)
    plt.axis([0, 400, 0, 80])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(3)
    plt.axis([0, 400, 0, 50])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel.png')
    plt.clf()
    plt.close()

    plt.figure(4)
    plt.axis([0, 400, 0, 250])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(5)
    plt.axis([0, 400, 0, 1.2])
    plt.axhline(y=1.0, color='black', linestyle='-.')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
    plt.legend()
    plt.savefig(model_dir+'response_parallel.png')
    plt.clf()
    plt.close()