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


def field_to_human(field):
    dd = dict()
    dd['bandwidths'] = ("Bitrate", lambda b: bytes_to_human0(b, speed=True, tobits=True))
    dd['copys'] = ("Memcpy Time", lambda b: time_to_human0(b))
    dd['waits'] = ("MPI_Wait Time", lambda b: time_to_human0(b))
    dd['posts'] = ("MPI_Send/Recv Time", lambda b: time_to_human0(b))
    dd['totals'] = ("Total Time", lambda b: time_to_human0(b))
    if not field in dd:
        return (field, lambda x:(x,"",1))
    return dd[field]

def time_to_human0(value, from_usec=True):
    units = ["Ksec","sec","ms","μs","ns","ps","fs"]
    factor = 1
    factor_step = 1000
    junit = 1
    if from_usec:
        junit = 3

    while value*factor > 1100:
        factor = factor/factor_step
        junit -= 1
    while value*factor < 0.900:
        factor = factor*factor_step
        junit += 1
    return (value*factor, units[junit], factor)

def bytes_to_human0(value, speed=False, tobits=False):
    units =         ["TiB",  "GiB",  "MiB",  "KiB",  "Byte"]
    speed_units =   ["TB/s", "GB/s", "MB/s", "KB/s", "B/s"]
    bitrate_units = ["Tbps", "Gpbs", "Mbps", "Kbps", "bps"]
    bit_units =     ["Tib",  "Gib",  "Mib",  "Kib",  "bits"]

    factor_step = 1024
    if speed: factor_step = 1000

    junit = 4
    factor = 1
    if (tobits): factor = 8

    while value*factor > 1100:
        factor = factor/factor_step
        junit -= 1
    while value*factor < 0.900:
        factor = factor*factor_step
        junit += 1

    if speed and tobits:    units = bitrate_units
    elif speed:             units = speed_units
    elif tobits:            units = bit_units

    return (value*factor, units[junit], factor)

def bytes_to_human(bytes):
    return "{0:.2f}{1}".format(*bytes_to_human0(bytes))

def bandwidth_to_human_bitrate(bytes_per_sec):
    return "{0:.2f}{1}".format(*bytes_to_human0(bytes,speed=True,tobits=True))


def load_pipeline_file(filename):
    data = json.load(open(filename))
    nranks = data['nranks']
    ntrials = data['ntrials']

    data['pp'] = {}
    data['pp']['full_filename'] = filename

    # each result is nranks * ntrials
    for jres,res in enumerate(data['results']):
        for field in ['bandwidths', 'copys', 'waits', 'posts', 'totals', 'jranks', 'jtrials']:
            res[field] = numpy.array(res[field]).reshape( (nranks, ntrials) )
    return data

def plot_res_vs_rank(res,dat,fieldname,fmt='b.',typical_val=None):
    nranks = dat['nranks']
    ntrials = dat['ntrials']
    (display_name, convertor) = field_to_human(fieldname)
    if typical_val is None:
        typical_val = numpy.median(res[fieldname])
    (_,display_units, display_factor) = convertor(typical_val)
    for jtrial in range(ntrials):
        xax = numpy.arange(nranks) + 0.5 * (jtrial/ntrials) - 0.25
        plt.plot(xax,display_factor*res[fieldname][:,jtrial],fmt,markersize=3, markeredgecolor='none',label='total')
    msgsz = "{0} Message".format(bytes_to_human(dat['message_size']))
    bufsz = "{0}x{1} Buffers".format(res['buffer_depth'],bytes_to_human(res['buffer_size']))
    plt.title(f"{fieldname} {msgsz} {bufsz}")
    plt.xlabel("Rank - #")
    plt.ylabel(f"{display_name} - {display_units}")

def plot_res_summary_vs_rank(res, dat):
    plt.figure()
    plt.subplot(121)
    plot_res_vs_rank(res,dat,'copys','b.',2000)
    plot_res_vs_rank(res,dat,'posts','g.',2000)
    plot_res_vs_rank(res,dat,'waits','r.',2000)
    plot_res_vs_rank(res,dat,'totals','k.',2000)
    plt.subplot(122)
    plot_res_vs_rank(res,dat,'bandwidths','r.')

def plot_res_vs_trial(res,dat,fieldname):
    nranks = dat['nranks']
    ntrials = dat['ntrials']
    (display_name, convertor) = field_to_human(fieldname)
    (_,display_units, display_factor) = convertor(numpy.median(res[fieldname]))
    for jrank in range(nranks):
        xax = numpy.arange(ntrials) + 0.5 * (jrank/nranks) - 0.25
        plt.plot(xax,display_factor*res[fieldname][jrank,:],'b.',markersize=3, markeredgecolor='none',label=display_name)
    msgsz = "{0} Message".format(bytes_to_human(dat['message_size']))
    bufsz = "{0}x{1} Buffers".format(res['buffer_depth'],bytes_to_human(res['buffer_size']))
    plt.title(f"{fieldname} {msgsz} {bufsz}")
    plt.xlabel("Trial - #")
    plt.ylabel(f"{display_name} - {display_units}")

def bar_res_vs_bufsize(dat, fieldname):
    nbuf_sizes = dat['nbuf_sizes']
    ndepths = dat['ndepths']
    fig, ax = plt.subplots(layout='constrained')

    xpos = numpy.arange(nbuf_sizes)
    width = (1/(ndepths+1))
    multiplier=0

    reduced_values = numpy.zeros( (nbuf_sizes, ndepths))
    for res in dat['results']:
        jsize = dat['buf_sizes'].index(res['buffer_size'])
        jdepth = dat['buf_depth'].index(res['buffer_depth'])
        reduced_values[jsize,jdepth] = numpy.mean(res[fieldname])

    (display_name, convertor) = field_to_human(fieldname)
    (_,display_units, display_factor) = convertor(reduced_values.mean())
    if fieldname == "bandwidths":
        display_factor *= dat['nranks']/2
        display_name = "Node-wide " + display_name
    reduced_values *= display_factor

    for jdepth in range(ndepths):
        offset = width * jdepth
        buf_depth = dat['buf_depth'][jdepth]
        rects = ax.bar(xpos + offset, reduced_values[:,jdepth], width, label=f"{buf_depth}-deep")

    ax.set_ylabel(f"{display_name} - {display_units}")
    ax.set_xticks(xpos + width, [ bytes_to_human(b) for b in dat['buf_sizes']])
    ax.legend(loc='lower right', ncols=3)
    msgsz = "{0} Message".format(bytes_to_human(dat['message_size']))
    rnkz = f"{dat['nranks']} Ranks"
    plt.title(f"{fieldname} {msgsz} {rnkz}")


def bar_res_vs_tmpspace(dat, fieldname):
    nbuf_sizes = dat['nbuf_sizes']
    ndepths = dat['ndepths']
    fig, ax = plt.subplots(layout='constrained')

    space_ax = []
    for buf_size in dat['buf_sizes']:
        for buf_depth in dat['buf_depth']:
            space_ax.append(buf_size*buf_depth)
    space_ax = list(set(space_ax))
    nspaces = len(space_ax)

    xpos = numpy.arange(nspaces)
    width = (1/(dat['ndepths']+1))
    multiplier=0

    reduced_values = numpy.zeros( (nspaces, ndepths))
    for res in dat['results']:
        space = res['buffer_size']*res['buffer_depth']
        jspace = space_ax.index(space)
        jdepth = dat['buf_depth'].index(res['buffer_depth'])
        reduced_values[jspace,jdepth] = numpy.mean(res[fieldname])

    (display_name, convertor) = field_to_human(fieldname)
    (_,display_units, display_factor) = convertor(reduced_values[reduced_values!=0].mean())
    if fieldname == "bandwidths":
        display_factor *= dat['nranks']/2
        display_name = "Node-wide " + display_name
    reduced_values *= display_factor

    for jdepth in range(ndepths):
        offset = width * jdepth
        buf_depth = dat['buf_depth'][jdepth]
        rects = ax.bar(xpos + offset, reduced_values[:,jdepth], width, label=f"{buf_depth}-deep")

    ax.set_ylabel(f"{display_name} - {display_units}")
    ax.set_xticks(xpos + width, [ bytes_to_human(b) for b in space_ax], rotation=90)
    ax.set_xlabel("Total bounce buffer space per rank")
    ax.legend(loc='lower right', ncols=3)
    msgsz = "{0} Message".format(bytes_to_human(dat['message_size']))
    rnkz = f"{dat['nranks']} Ranks"
    plt.title(f"{fieldname} {msgsz} {rnkz}")

print("Module re-loaded")