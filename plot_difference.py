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
   bp.figure.savefig('stripplot.png')

   
@default_style
def cumulative_plot(df):
   df = df.mean().apply(np.sqrt)
   df = df.sort_index(ascending=False)
   p = df.plot(marker='o')
   plt.xlabel('lnrhoref')
   plt.ylabel('sqrt{(q_con - q_uncon)^2 /N} [e]')
   p.figure.savefig('cumulative.png')


@default_style
def main():
   print('Reading Data ...')
   df_con = pd.read_csv('constrained_vs_rhoref.csv', decimal='.')
   df_uncon = pd.read_csv('unconstrained_vs_rhoref.csv', decimal='.')
   dfs = [df_con, df_uncon]
   for df in dfs:
      df = df.set_index(['residue', 'atom'])
      df = df.astype(float)

   diff_df = pd.DataFrame(index=df_con.index)
   diff_df['atom'] = df_con.atom
   diff_df['residue'] = df_con.residue
   for i in range(-3, -10, -1):
      #diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)]).abs()
      diff_df[str(i)] = (df_con[str(i)] - df_uncon[str(i)])**2
   df = diff_df

   print('Plotting ...')
   stripplot(df)
   cumulative_plot(df)
   print('Done.')
   

if __name__ == '__main__':
	main()
