import sqlite3
import matplotlib.pyplot as plt
import numpy as np

def get_value_sums(database_str):
    conn = sqlite3.connect(database_str)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()
    conn.commit()
    #conn.close()
    table_names =  [i[0] for i in tables]
    d = {}
    for name in table_names:
        #print(name)
        try:
            cur.execute("SELECT value FROM {} ORDER BY STEP".format(name))
            rows = cur.fetchall()
            times = [item[0] for item in rows]
            times = np.array(times)
            d[name] = times.sum()
        except:
            pass
    #print(d)
    return d


def get_elwise_linear_times(database_str):
    conn = sqlite3.connect(database_str)#("vortex.sqlite-unoptimized")
    cur = conn.cursor()
    cur.execute("SELECT value FROM diff_time ORDER BY step")
    rows = cur.fetchall()

    times = [item[0] for item in rows]
    #for i in rows:
    #    print(i)

    cur.close()
    conn.commit()
    conn.close()
    
    return times

udict = get_value_sums("lump.sqlite-unoptimized_special")
odict = get_value_sums("lump.sqlite-optimized_specialv7")
keys = udict.keys()
rdiff_dict = {}
usum = 0.0
osum = 0.0
for key in keys:
    if "time"in key:
        uval = udict[key]
        oval = odict[key]
        usum += uval
        osum += oval
        try: 
            rel_diff = (oval - uval)/uval
            print("{}: {} {} {}".format(key, uval, oval, rel_diff))
            #print(rel_diff)
            if rel_diff != 0 or rel_diff != np.nan:
                #print(rel_diff)
                rdiff_dict[key] = rel_diff
        except:
            pass

print("{} {}".format("Total", (osum - usum)/usum))
rdiff_dict["Total"] = (osum - usum)/usum

values = rdiff_dict.values()
titles = rdiff_dict.keys()
y_pos = np.arange(len(titles))
plt.bar(y_pos, values, align='center')
plt.xticks(y_pos, titles, rotation=-90)
plt.ylabel("Relative change (negative is faster)")
plt.title("Relative change in kernel run times for 3D lump case, order 3")


plt.tight_layout()
"""
unopt_times = get_elwise_linear_times("lump.sqlite-unoptimized_special")
opt_times = get_elwise_linear_times("lump.sqlite-optimized_special")
plt.xlabel("Step number")
plt.ylabel("Time")
plt.title("diff times on 3D Lump case")

plt.plot(unopt_times, label="Unoptimized")
plt.plot(opt_times, label="Optimized")
plt.legend()
"""
plt.show()
