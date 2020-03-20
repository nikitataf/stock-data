from matplotlib import cycler
import matplotlib as mpl


def matplotlib_nikita_style():
    """Matplotlib style by Nikita Tafintsev. Applies to all figures that will be created"""
    mpl.rc('figure', facecolor='1',
           autolayout=False,  # When True, automatically adjust subplot parameters to make the plot fit the figure
           titleweight='normal',  # weight of the figure title
           figsize=(8, 6))  # figure size in inches)
    mpl.rc('figure.subplot',
           left=.12, bottom=.11, right=0.97, top=.95)
    mpl.rc('axes', edgecolor='(0.04, 0.14, 0.42)',
           facecolor='(0.96, 0.98, 0.99)',
           labelsize='medium',
           # labelweight='bold',
           # labelcolor='(0.04, 0.14, 0.42)',
           grid='True')
    # mpl.rc('text', color='(0.04, 0.14, 0.42)')
    mpl.rc('lines', linewidth=3, markersize=7)
    mpl.rc('font', family='Sans', size=15)
    mpl.rc('legend', fontsize='medium', facecolor=(0.96, 0.98, 0.99))
    mpl.rc('grid', linestyle='--')
    mpl.rc('axes', prop_cycle=cycler(color=['#006BB2', '#B22400', 'green', '6E666D', '000000', 'chocolate']))
    mpl.rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
