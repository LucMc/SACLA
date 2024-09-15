import click
from prob_lyap.eval.vector_field_plot import create_vectorfield
from prob_lyap.objectives import OBJ_TYPES
import pandas as pd
import matplotlib.pyplot as plt

def get_obj_data(dfs, objective):
    x = []
    y = []
    std = []
    for step, df in dfs.items():
        x.append(step)
        y.append(df.loc[objective]["percent_mean"])
        std.append(df.loc[objective]["percent_std"])
    return x, y, std

def lsac_objs(lsac_file, plot_file):
    bound = 240_000
    df = pd.read_csv(f"./data/{lsac_file}.csv")
    df['Objective'] = df['Objective'].replace('mixed_adv', 'SACLA')
    df['Objective'] = df['Objective'].replace('standard', 'Vanilla SAC')
    df = df[df['step'] <= bound]
    plt.figure(figsize=(10,6), dpi=150)



    dfs_by_step = {step: group for step, group in df.groupby('step')}
    updated_dfs = {}
    for step, df in dfs_by_step.items():

        result = df.groupby('Objective').agg({
        'percent': ['mean', 'std'],
        '+ive': ['mean', 'std'],
        '-ive': ['mean', 'std']
        })

        result.columns = ['_'.join(col).strip() for col in result.columns.values]
        updated_dfs[step] = result

    for objective in dfs_by_step[20_000].Objective.unique():
        data = get_obj_data(updated_dfs, objective=objective)
        # breakpoint()
        plt.errorbar(data[0], data[1], yerr=data[2], label=objective)

    total = dfs_by_step[20_000]["+ive"] + dfs_by_step[20_000]["-ive"]
    print("total in grid", total.mean())
    breakpoint()
    # plt.axhline(y=int(total.mean()), linestyle="--", label="total")
    plt.xlabel("training time step")
    plt.ylabel("Percent -ive Lie derivatives")
    plt.legend()
    plt.savefig(f"{plot_file}_lsac.png")
    plt.close()

def polyc_vs_lsac(polyc_file, lsac_file, plot_file):
    lsac_df = pd.read_csv(f"./data/{lsac_file}.csv")
    lsac_df = lsac_df[lsac_df["Objective"] == "mixed_adv"]
    
    polyc_df = pd.read_csv(f"./data/{polyc_file}.csv")
    plt.figure(figsize=(10,6), dpi=150)


    dfs = {
        "POLYC": polyc_df,
        "LSAC": lsac_df
    }

    for alg, df in dfs.items():
        dfs_by_step = {step: group for step, group in df.groupby('step')}
        updated_dfs = {}
        for step, df in dfs_by_step.items():

            result = df.groupby('Objective').agg({
            'percent': ['mean', 'std'],
            '+ive': ['mean', 'std'],
            '-ive': ['mean', 'std']
            })
            
            result.columns = ['_'.join(col).strip() for col in result.columns.values]
            if alg == "lsac":
                updated_dfs[step]= result.loc["mixed_adv"]
            else:
                updated_dfs[step]= result 
        
        print([x["percent_mean"] for x in updated_dfs.values()])
        if alg == "POLYC":
            k = "mixed_adv" # Quick fix
            plt.errorbar(list(updated_dfs.keys()), [x.loc[k]["percent_mean"] for x in updated_dfs.values()], yerr=[x.loc[k]["percent_std"] for x in updated_dfs.values()], label="POLYC")
        elif alg == "LSAC":
            plt.errorbar(list(updated_dfs.keys()), [x.loc["mixed_adv"]["percent_mean"] for x in updated_dfs.values()], yerr=[x.loc["mixed_adv"]["percent_std"] for x in updated_dfs.values()], label="SACLA")


    total = dfs_by_step[20_000]["+ive"] + dfs_by_step[20_000]["-ive"]
    print("total in grid", total.mean())
    # plt.axhline(y=int(total.mean()), linestyle="--", label="total")
    plt.xlabel("training time step")
    plt.ylabel("Percent -ive Lie derivatives")
    plt.legend()
    plt.savefig(f"{plot_file}_POLYC.png")
    plt.close()

@click.command()
@click.option("-a", "--algorithm", default="lsac", type=str, help="Which plot to show")
@click.option("--polyc-file", type=str, default="new_polyc_data1", help="Name of POLYC data from lie vector field experiment")
@click.option("--lsac-file", type=str, default="lsac_data", help="Name of POLYC data from lie vector field experiment")
@click.option("--plot-file", type=str, default="lvf", help="File name for plot")
def main(algorithm: str,
         polyc_file: str,
         lsac_file: str,
         plot_file: str):
    
    if algorithm == "polyc":
        polyc_vs_lsac(polyc_file, lsac_file, plot_file)
    elif algorithm == "lsac":
        lsac_objs(lsac_file, plot_file)

if __name__ == "__main__":
    # lsac_objs()
    main()
 