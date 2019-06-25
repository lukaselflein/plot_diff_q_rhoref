"""Plot long format table of point charges.
Copyright 2019 Simulation Lab
University of Freiburg
Author: Lukas Elflein <elfleinl@cs.uni-freiburg.de>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
def stripplot(df):
   df = pd.melt(df, id_vars=['atom', 'residue'])
   df['variable'] = df['variable'].astype(int)
   bp = sns.swarmplot(x='value', y='atom', data=df, hue='variable', 
                      palette=sns.color_palette("coolwarm", len(df.variable.unique())))
   bp.figure.savefig('img/stripplot.png')

   
@default_style
def cumulative_plot(df, plotname='cumulative.png'):
   p = plt.plot(df.index, df.values, marker='o')
   print(df.index.tolist())
   print(df.values)
   plt.xlabel('lnrhoref')
   plt.ylabel('sqrt{(q_con - q_uncon)^2 /N} [e]')
   plt.savefig('img/' + plotname)

def parse_data(constr_file, unconstr_file):
   print('Reading files {}, {}'.format(constr_file, unconstr_file))
   df_con = pd.read_csv('data/' + constr_file, decimal='.')
   df_uncon = pd.read_csv('data/' + unconstr_file, decimal='.')

   dfs = [df_con, df_uncon]
   for df in dfs:
      df.columns = df.columns.str.replace('q_unconstrained_', '-')
      df = df.set_index(['residue', 'atom'])
      df = df.astype(float)

   diff_df = pd.DataFrame(index=df_con.index)
   diff_df['atom'] = df_con.atom
   diff_df['residue'] = df_con.residue
   for i in range(-3, -10, -1):
      #diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)]).abs()
      diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)])**2
   return diff_df


def calc_diff(df):
   df = df.mean().apply(np.sqrt)
   df = df.sort_index(ascending=False)
   return df

@default_style
def compare_plot(df, plotname='cumulative.png'):
   print('Plotting {}'.format(plotname))
   p = sns.pointplot(x='lnrho', y='diff', hue='q', data=df)
   ax = p.axes
#   ax.set_ylim(df['diff'].min() - 0.05, 0.4)
   #plt.xlabel('lnrhoref')
   plt.ylabel('sqrt{(q_con - q_uncon)^2 /N} [e]')
   p.figure.savefig('img/' + plotname)


@default_style
def main():
   # Zero Charge
   df_0 = parse_data('constrained_vs_rhoref_q0.csv', 'unconstrained_vs_rhoref_q0.csv')
   diff_0 = calc_diff(df_0) 
   cumulative_plot(diff_0, 'cumulative_q0.png')

   # Charge = 1
   df_1 = parse_data('constrained_vs_rhoref.csv', 'unconstrained_vs_rhoref.csv')
   diff_1 = calc_diff(df_1) 
   cumulative_plot(diff_1, 'cumulative_q1.png')

   # Charge = 2
   df_2 = parse_data('constrained_vs_rhoref_q2.csv', 'unconstrained_vs_rhoref_q2.csv')
   diff_2 = calc_diff(df_2) 
   cumulative_plot(diff_2, 'cumulative_q2.png')

   # Combine everything into one dataframe
   df = pd.DataFrame(index=diff_1.index)
   df['0'] = diff_0.values
   df['1'] = diff_1.values
   df['2'] = diff_2.values
   df['lnrho'] = df.index
   df = df.melt(id_vars=['lnrho'], var_name='q', value_name='diff')
   df['q'] = df['q'].astype(int)
   df['lnrho'] = df['lnrho'].astype(int)
   compare_plot(df, plotname='lnrho_q_comparison.png')

   print('Done.')
   

if __name__ == '__main__':
	main()
