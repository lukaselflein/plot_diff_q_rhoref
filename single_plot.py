"""Plot long format table of point charges.
Copyright 2019 Simulation Lab
University of Freiburg
Author: Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from plot_difference import calc_diff

def default_style(func):
	"""A decorator for setting global plotting styling options."""
	def wrapper(*args, **kwargs):
		fig = plt.figure(figsize=(16,10))
		sns.set_context("talk", font_scale=0.9)
		plt.tick_params(grid_alpha=0.2)
		func(*args, **kwargs)
		plt.clf()
	return wrapper
   
@default_style
def cumulative_plot(df, plotname='cumulative.png'):
   p = plt.plot(df.index, df.values, marker='o')
   print(df.index.tolist())
   print(df.values)
   plt.xlabel('lnrhoref')
   plt.ylabel('sqrt{(q_con - q_uncon)^2 /N} [e]')
   plt.savefig(plotname)

def calc_diff(df):
   df = df.mean().apply(np.sqrt)
   df = df.sort_index(ascending=False)
   return df

def parse_data(constr_file, unconstr_file):
   print('Reading files {}, {}'.format(constr_file, unconstr_file))
   df_con = pd.read_csv(constr_file, decimal='.')
   df_uncon = pd.read_csv(unconstr_file, decimal='.')

   dfs = [df_con, df_uncon]
   for df in dfs:
      df.columns = df.columns.str.replace('q_unconstrained_', '-')
      df = df.set_index(['residue', 'atom'])
      df = df.astype(float)

   diff_df = pd.DataFrame(index=df_con.index)
   diff_df['atom'] = df_con.atom
   diff_df['residue'] = df_con.residue
   lnrho_range = [int(i) for i in df_con.columns if i[1:].isdigit()]
   for i in lnrho_range: # range(-3, -10, -1):
      #diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)]).abs()
      diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)])**2
   return diff_df


def cmd_parser():
	parser = argparse.ArgumentParser(prog='',
					 description='Plot difference between constrained and unconstrained charges.')

	parser.add_argument('-con', metavar='constrained.csv',
        help='The location of the contstrained charge table, in the .csv format.') 

	parser.add_argument('-un', metavar='unconstrained.csv',
        help='The location of the unconstrained charge table, in the .csv format.') 
	
	parser.add_argument('-png', metavar='plot.png',
        help='The location where the plot should be saved.',
	default='plot.png')

	args = parser.parse_args()

	return args.con, args.un, args.png

def main():
   # Zero Charge

   constrained_file, unconstrained_file, png_file = cmd_parser()

   print('Reading data ...')
   df = parse_data(constrained_file, unconstrained_file)
   diff = calc_diff(df) 
   print('Plotting ...')
   cumulative_plot(diff, png_file)

   print('Done.')
   

if __name__ == '__main__':
	main()
