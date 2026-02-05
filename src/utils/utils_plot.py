import matplotlib.pyplot as plt
import seaborn as sns


# Set plot parameters.
sns.set_theme(
    context='paper',
    style='ticks',
    palette='deep',
    font='sans-serif',
    font_scale=1,
    rc={
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',  # Ensures text is editable in SVGs
        # 'svg.embed_char_paths': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        # Font sizes
        'axes.labelsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.titlesize': 6,
        'legend.fontsize': 6,
        'font.size': 6,
        'figure.titlesize': 6,
        'figure.labelsize': 6,
        'savefig.facecolor': 'white',
        'axes.unicode_minus': False,  # Ensure minus signs are not used for tick marks
    }
)

# # Color palettes.
# reward_palette = sns.color_palette(['#c959affe', "#1ebe8e"])
# reward_palette_r = sns.color_palette([ '#1b9e77', '#c959affe'])
# cell_types_palette = sns.color_palette(['#8c8c8c', '#1f77b4', '#ff9600ff'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# # s2_m1_palette = sns.color_palette(['#6D9BC3', '#E67A59'])
# s2_m1_palette = sns.color_palette(['steelblue','salmon'])
# # s2_m1_palette = sns.color_palette(['#ff9600ff','#4682b4',]) 
# stim_palette = sns.color_palette(['#1f77b4', '#ff9600ff', '#333333'])
# behavior_palette = sns.color_palette(['#06fcfeff', '#1f77b4', '#c959affe', '#1b9e77', '#8c8c8c', '#333333'])
# trial_type_rew_palette = sns.color_palette(['#06fcfeff', '#1f77b4', '#90ee90', '#1b9e77', '#8c8c8c', '#333333'])  # auditory misses, auditory hits, whisker misses, whisker hits, correct rejection, false alarm
# trial_type_nonrew_palette = sns.color_palette(['#06fcfeff', '#1f77b4', '#dda0dd', '#c959affe', '#8c8c8c', '#333333'])  # auditory misses, auditory hits, whisker misses, whisker hits, correct rejection, false alarm

# Color palettes (more saturated for better distinguishability).
reward_palette = sns.color_palette(['#D656AE', '#1BC477'])
reward_palette_r = sns.color_palette(['#1BC477', '#D656AE'])
cell_types_palette = sns.color_palette(['#8c8c8c', '#2657CC', '#ff9600ff'])  # medium grey, vivid blue, bright orange
s2_m1_palette = sns.color_palette(['#2657CC', '#ff9600ff'])  # saturated steel blue, vibrant red-orange
stim_palette = sns.color_palette(['#2657CC', '#ff9600ff', '#222222'])  # saturated blue, vivid orange, dark neutral
behavior_palette = sns.color_palette(['#0dddddff', '#2657CC', '#D656AE', '#1BC477', '#8c8c8c', '#222222'])
trial_type_rew_palette = sns.color_palette(['#0dddddff', '#2657CC', "#94F5A5", '#1BC477', '#8c8c8c', '#222222'])
trial_type_nonrew_palette = sns.color_palette(['#0dddddff', '#2657CC', "#FFA2E3", '#D656AE', '#8c8c8c', '#222222'])

# # Color palettes.
# reward_palette = sns.color_palette(['#980099ff', '#009600ff'])
# reward_palette_r = sns.color_palette([ '#009600ff', '#980099ff'])
# cell_types_palette = sns.color_palette(['#807f7fff', '#ca59afff', '#0100fdff'])  # Grey for all cells, Blue for S2 projection, Orange for M1 projection
# s2_m1_palette = sns.color_palette(['#ca59afff', '#0100fdff'])
# # s2_m1_palette = sns.color_palette(['#ff9600ff','#4682b4',]) 
# stim_palette = sns.color_palette(['#0100fdff', '#ff9600ff', '#010101ff'])
# behavior_palette = sns.color_palette(['#06fcfeff', '#0100fdff', '#980099ff', '#009600ff', '#807f7fff', '#010101ff'])
# trial_type_rew_palette = sns.color_palette(['#06fcfeff', '#0100fdff', '#66c266', '#009600ff', '#807f7fff', '#010101ff'])  # auditory misses, auditory hits, whisker misses, whisker hits, correct rejection, false alarm
# trial_type_nonrew_palette = sns.color_palette(['#06fcfeff', '#0100fdff', '#e699e6ff', '#980099ff', '#807f7fff', '#010101ff'])  # auditory misses, auditory hits, whisker misses, whisker hits, correct rejection, false alarm

# Mice groups.
mice_groups = { 
    'gradual_day0': ['GF278', 'GF301', 'GF305', 'GF306', 'GF313', 'GF317', 'GF318', 'GF323', 'GF328', ],
    'step_day0': ['GF339', 'AR176', ],
    'psth_mapping_increase': ['GF305', 'GF306', 'GF308', 'GF313', 'GF318', 'GF323', 'GF334',],
    'good_day0':['GF240', 'GF241','GF248','GF253','GF267','GF278','GF257','GF287',
                 'GF261','GF266','GF300','GF301','GF303','GF305','GF306',
                 'GF307','GF308','GF310','GF311','GF313','GF314',
                 'GF317','GF318','GF323','GF325','GF326','GF327','GF328','GF334','GF336','GF337','GF338','GF339',
                 'GF353','GF354','GF355','MI023','MI026','MI023','MI028','MI029','MI030','MI031',
                 'MI054','MI055','AR121','AR133','AR135','AR176',],
    'meh_day0': ['GF252', 'GF256','GF272','GF264','GF333','AR123','AR143','AR177','AR127',],
    'bad_day0': ['GF271','GF290','GF291','GF292','GF293','GF249','MI012','MI014','MI027','MI039',
                 'MI040','MI044','MI045','MI053','AR115','AR116','AR117','AR119','AR120','AR122','AR144',
                 'AR163',],
}