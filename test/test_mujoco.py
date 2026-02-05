import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.new_td3_main import main, code_creation
from src.utils import get_arguments_mujoco

if __name__=='__main__':
    
    args = get_arguments_mujoco().parse_args()

    # Access hyperparameters in your script
    use_rnd = args.rnd
    domain = args.dom
    timesteps = args.steps
    ade = args.ade
    max_episode_steps = timesteps
    noisy_nets = args.noisy_nets
    mod_rnd = args.mod_rnd
    per = args.per
    ucb = args.UCB
    rho = args.rho
    adeu_ensemble = args.adeu_ensemble
    include_pos = args.include_pos

    comb = False
    horizon = int(1.2e6)
    

        

    seeds = 20
    mean_evaluation_rewards = []
    code = code_creation(adaptive_exploration=ade, 
                                           ucb = ucb, 
                                           use_rnd=use_rnd, 
                                           noisy_nets=noisy_nets, 
                                           comb=comb, 
                                           per=per, 
                                           mod_rnd=mod_rnd, 
                                           adeu_ensemble=adeu_ensemble,
                                           rho = rho
    )
    print(f"{domain} + code:{code}")

    for s in range(seeds):
        main(seed = s, 
             horizon=horizon, 
             timesteps=timesteps, 
             ucb = ucb, 
             domain=domain, 
             ade = ade, 
             use_rnd=use_rnd, 
             noisy_nets=noisy_nets, 
             comb =comb, 
             per=per, 
             mod_rnd = mod_rnd,
             include_pos = include_pos,
             adeu_ensemble = adeu_ensemble,
             rho = rho
        )
    

    