import matplotlib
import numpy
import json
import matplotlib.pyplot as plt

def load_file(filename):
    data = json.load(open(filename))
    try:
        nranks = data['test_config']['nranks']
        ntrials = data['test_config']['ntrials']
    except KeyError:
        nranks = len(data['test_result'][0]['completion_time_avg'])
        ntrials = len(data['test_result'])
        data['test_config']['nranks'] = nranks
        data['test_config']['ntrials'] = ntrials

    data['pp'] = {}
    data['pp']['full_filename'] = filename
    data['pp']['filename'] = filename.split('/')[-1].replace('.dat','')
    data['pp']['title'] = data['pp']['filename'].replace('_',' ')
    data['pp']['total_mat'] = numpy.zeros( (nranks, ntrials) )
    for jres,res in enumerate(data['test_result']):
        data['pp']['total_mat'][:,jres] = res['total_time_avg']
    return data
def scatter_time_v_rank(fname):
    data = load_file(fname)
    plt.figure()
    nresults = len(data['test_result'])
    nranks = len(data['test_result'][0]['completion_time_avg'])
    h = [0,1,2]
    for jres,res in enumerate(data['test_result']):
        xax = numpy.arange(nranks) + 0.5 * (jres/nresults) - 0.25
        h[0], = plt.plot(xax,res['submit_time_avg'],'rs',markersize=3, markeredgecolor='none',label='submit')
        h[1], = plt.plot(xax,res['completion_time_avg'],'b^',markersize=3, markeredgecolor='none',label='complete')
        h[2], = plt.plot(xax,res['total_time_avg'],'g.',label='total',markersize=3)
    plt.plot(numpy.arange(nranks), numpy.median(data['pp']['total_mat'], axis=1), 'ko',markersize=6,markeredgecolor='w')
    plt.ylabel('Time - usec')
    plt.xlabel('MPI Rank')
    plt.title(data['pp']['title'])
    plt.gca().legend(handles=h,labels=['submit','complete','total'])
    plt.show()
def scatter_time_v_rank2(fname):
    data = load_file(fname)
    xax = numpy.arange(data['test_config']['nranks'])
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
    root = data['pp']['total_mat'][0,:];
    others = data['pp']['total_mat'][1:,:].reshape( (1,-1) ).T
    plt.ylabel('Time - usec')
    axs[0].violinplot(root)
    axs[1].violinplot(others)
    axs[0].set_title('Root')
    axs[1].set_title('Others')
    fig.suptitle(fname)


    # plt.xlabel('MPI Rank')
    # fig.title(fname)
#    plt.show()

def compare_roots(fnames):
    fnames_nodat = [ f.split('/')[-1].replace(".dat","") for f in fnames ]
    common_set = set.intersection( *[ set([t for t in n.split('_')]) for n in fnames_nodat ] )
    short_names = [ '_'.join(t for t in n.split('_') if t not in common_set) for n in fnames_nodat]
    common_name = '_'.join(t for t in fnames_nodat[0].split('_') if t in common_set)

    data_sets = [ load_file(f) for f in fnames]
    # xax = numpy.arange(data['test_config']['nranks'])
    fig, axs = plt.subplots(nrows=1, ncols=1, sharey=True)
    axs = [axs]
    roots = [ dat['pp']['total_mat'][0,:] for dat in data_sets ]
    others = [ dat['pp']['total_mat'][1:,:].ravel() for dat in data_sets ]
    axs[0].set_ylabel('Time - usec')
    # axs[0].set_yscale('log')
    # axs[1].set_ylabel('Time - usec')
    axs[0].violinplot(roots)
    # axs[1].violinplot(others)
    # axs[1].set_title('Other Latency')
    axs[0].set_title('Root Latency')
    tickx = numpy.arange(1,len(fnames)+1)
    axs[0].set_xticks(ticks=tickx, labels=short_names, rotation=45, ha='right')
    # axs[1].set_xticks(ticks=tickx, labels=short_names, rotation=45, ha='right')
    fig.subplots_adjust(bottom=0.25)
    fig.suptitle(common_name)


print("Module re-loaded")