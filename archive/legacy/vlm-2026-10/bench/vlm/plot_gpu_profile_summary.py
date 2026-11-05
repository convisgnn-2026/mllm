#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
 d=json.loads((root/'gpu_analysis.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4.5))
 for ax,role,solo,title in zip(axes,['llm','encoder'],['llm_only','vlm_only_high'],['Independent LLM','VLM encoder']):
  values=[d['cases'][c]['services'][role]['launch_end_to_kernel_start_us']['p90']/1000 for c in [solo,'mixed_high']]
  bars=ax.bar(['Only this service\nreceives requests','Both services\nreceive requests'],values,color=['#78909c','#e07a32'],width=.55)
  ax.bar_label(bars,labels=[f'{v:.2f} ms' for v in values],padding=5,fontsize=12)
  ax.set_title(title);ax.set_ylabel('Kernel launch return → GPU start, p90 (ms)');ax.set_ylim(0,max(values)*1.25);ax.spines[['top','right']].set_visible(False)
 fig.suptitle('GPU execution waits increase during shared execution',fontsize=14)
 fig.text(.5,.025,'Profiled diagnostic, seed 0. Includes own-stream queuing; not end-to-end request latency.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(root/'gpu_wait_summary.png',dpi=180);fig.savefig(root/'gpu_wait_summary.svg');plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
