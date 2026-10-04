"""Render numerical/exact Rosen packet movies from saved snapshots (no JAX)."""
import argparse
import json
import os
from pathlib import Path
import subprocess
os.environ.setdefault('MPLCONFIGDIR','/tmp/em-waves-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

MOVIES={
    'EM_propagation': [('E',r'Electric field $E^x$'),('B',r'Magnetic field $B^y$'),('rho',r'EM energy density $\rho$')],
    'gravitational_response': [('gamma_xx',r'Transverse metric $\gamma_{xx}=\gamma_{yy}$'),('W',r'Conformal factor $W$'),('K',r'Trace of extrinsic curvature $K$')],
}


def bounds(values):
    low,high=float(np.min(values)),float(np.max(values))
    pad=max((high-low)*.12,1e-12)
    return low-pad,high+pad


def render(source,output=None,fps=25):
    source=Path(source); output=Path(output) if output else source/'movies'
    output.mkdir(parents=True,exist_ok=True)
    data=np.load(source/'snapshots.npz')
    config=json.loads((source/'configuration.json').read_text())
    if config['status']!='complete': raise ValueError('Run has not completed')
    times=data['time']; z=data['z']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'#f7f9fc','axes.facecolor':'white'})
    results={}
    for name,fields in MOVIES.items():
        destination=output/(name+'.mp4')
        if destination.exists(): raise FileExistsError(destination)
        fig,axes=plt.subplots(3,2,figsize=(13,8),gridspec_kw={'width_ratios':[2.4,1]},layout='constrained')
        artists=[]
        for row,(key,label) in enumerate(fields):
            ax,err=axes[row]
            actual,reference=data[key],data[key+'_exact']
            numerical,=ax.plot(z,actual[0],color='#0868ac',lw=2,label='Numerical')
            analytic,=ax.plot(z,reference[0],color='#212121',ls='--',lw=1.3,label='Exact')
            ax.set(title=label,ylim=bounds(np.concatenate((actual,reference))),xlim=(z[0],z[-1]),ylabel='Geometric units')
            residual=actual-reference
            difference,=err.plot(z,residual[0],color='#b34a17',lw=1.5)
            limit=max(float(np.max(abs(residual)))*1.15,1e-13)
            err.set(title='Numerical minus exact',ylim=(-limit,limit),xlim=(z[0],z[-1]))
            err.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
            artists.append((key,numerical,analytic,difference))
        for ax in axes[-1]: ax.set_xlabel('Propagation coordinate z')
        fig.legend([artists[0][1],artists[0][2]],['Numerical','Exact'],
                   loc='outside lower center',ncol=2,frameon=False)
        title=fig.suptitle('',fontsize=17,fontweight='medium')
        def update(i):
            title.set_text(f"Rosen EM pp-wave  |  {'Electromagnetic propagation' if name=='EM_propagation' else 'Gravitational response'}\nt = {times[i]:.3f}   ·   N = {config['n']}   ·   propagation +z")
            for key,numerical,analytic,difference in artists:
                numerical.set_ydata(data[key][i]);analytic.set_ydata(data[key+'_exact'][i])
                difference.set_ydata(data[key][i]-data[key+'_exact'][i])
        writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-crf','18','-pix_fmt','yuv420p'],
                           metadata={'title':name,'comment':'Numerical first-order Einstein-Maxwell evolution; exact Rosen reference'})
        update(0)
        fig.canvas.draw()
        # Limits, labels, and title extent remain fixed throughout the movie.
        fig.set_layout_engine(None)
        with writer.saving(fig,str(destination),dpi=110):
            for i in range(len(times)):
                update(i);writer.grab_frame()
        # Standalone frames make numerical/visual review reproducible.
        for label,i in [('start',0),('middle',len(times)//2),('end',len(times)-1)]:
            update(i);fig.savefig(output/f'{name}_{label}.png',dpi=110)
        plt.close(fig)
        subprocess.run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                        '-show_entries','stream=width,height,nb_read_frames,duration','-of','json',str(destination)]))
        stream=probe['streams'][0]
        if int(stream['nb_read_frames'])!=len(times): raise RuntimeError('Video frame count mismatch')
        results[name]=dict(path=str(destination.resolve()),decode_passed=True,**stream)
        print(str(destination),flush=True)
    (output/'movies.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'output_validation'/'n512')
    p.add_argument('--output',type=Path)
    p.add_argument('--fps',type=int,default=25)
    a=p.parse_args()
    if a.fps<=0:p.error('--fps must be positive')
    render(a.input,a.output,a.fps)
