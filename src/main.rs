use std::cmp;
use std::fs::{File, remove_file};
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
use std::time::Instant;
use rand::Rng;
use hungarian::minimize;
use pathfinding::prelude::{kuhn_munkres_min, Matrix};

/*
https://discuss.python.org/t/on-macos-14-pip-install-throws-error-externally-managed-environment/50352/3
mkdir ~/.venv

python3 -m venv ~/.venv

#	Creates the following in ~/.venv
#		bin/
#		include/
#		lib/
#		pyvenv.cfg
		
#		does not create pip-selfcheck.json 

# to activate the venv
source ~/.venv/bin/activate

# now you can..
python3 -m pip install <module name>

# and it will install the module in the virtual env

# to deactivate the venv
deactivate # or exit the shell

# you should see the folder name for the venv below your prompt when active, like so
pstivers3@mbp ~/repos/learn/pythonlearn
$ 
(.venv) 

# Note, you can chose any folder location and name that you want for the venv. ~/.venv is typical.
# Your project code can be in any directory.
*/
const BIG_VALUE: u16 = 65000;
const MAX_ITER: usize = 1;

const GLPK:usize = 0;
const RUST:usize = 1;
const RUST2:usize = 2;
const C:usize = 3;
const CPP:usize = 4;

fn main()  -> std::io::Result<()> {
    let mut cost: [[u16; 1000]; 1000] = [[0; 1000]; 1000];
    let demand_size: usize = 30;
    let supply_size: usize = 40;
    let max_size: usize = cmp::max(demand_size, supply_size);
    let mut cost_vec: [Vec<u32>; 10] = [const { Vec::new() }; 10];
    let mut time_vec: [Vec<u64>; 10] = [const { Vec::new() }; 10];

    for iter in 0 .. MAX_ITER {
        init_cost(&mut cost, supply_size, demand_size);
        
        // --------------- GLPK ------------------
        let glpk: Vec<i16>;
        write_input_balanced("input.txt", max_size, &cost);

        let mut start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("python3 glpk.py").output().expect("failed to execute process");
        time_vec[GLPK].push(start.elapsed().as_secs());

        let glpk_cost;
        (glpk_cost, glpk) = read_results_binary("output.txt", max_size, &cost);
        cost_vec[GLPK].push(glpk_cost);
        println!("GLPK ({}): {:?}", glpk_cost, glpk);

        // ----------------- RUST ----------------------
        // https://crates.io/crates/hungarian
        start = Instant::now();
        let munk = munkres(&cost, supply_size, demand_size);
        
        time_vec[RUST].push(start.elapsed().as_secs());
        let munk_cost: u32 = sum_up_cost(&munk, &cost);
        cost_vec[RUST].push(munk_cost);
        println!("Munkres ({}): {:?}", munk_cost, munk);

        // ------------------ RUST faster ------------------
        // https://crates.io/crates/pathfinding/4.3.1
        // !! "number of rows must not be larger than number of columns"
        start = Instant::now();
        let (_, ret) = munkres2(&cost, max_size, max_size);
        time_vec[RUST2].push(start.elapsed().as_secs());
        
        let mut munk2_cost = 0;
        for (s, d) in ret.iter().enumerate() {
            if cost[s][*d as usize] < BIG_VALUE {
                munk2_cost += cost[s][*d as usize] as u32;
            }
        }
        cost_vec[RUST2].push(munk2_cost);
        println!("Munkres2 ({}): {:?}", munk2_cost, ret);

        // ---------------- C ------------------------
        // https://github.com/xg590/munkres
   /*     write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("./munkres1").output().expect("failed to execute process");
        time_vec[C].push(start.elapsed().as_secs());
        let (c_cost, c_ret) = read_results_index("output.txt", &cost);
        println!("C ret ({}) {:?}", c_cost, c_ret);
*/
        // ---------------- C++ ------------------------
        // https://github.com/mcximing/hungarian-algorithm-cpp
        write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("./munkres2").output().expect("failed to execute process");
        time_vec[CPP].push(start.elapsed().as_secs());
        let (c_cost, c_ret) = read_results_index("output.txt", &cost);
        println!("CPP ret ({}) {:?}", c_cost, c_ret);
        cost_vec[CPP].push(c_cost);


        // --------------- COMPARING RESULTS
        if glpk_cost != munk_cost {
            println!("Cost does not equal, GLPK: {}, Munkres: {}", glpk_cost, munk_cost);
        }
        if glpk_cost != munk2_cost as u32 {
            println!("Cost does not equal, GLPK: {}, Munkres2: {}", glpk_cost, munk2_cost);
        }
        if glpk_cost != c_cost as u32 {
            println!("Cost does not equal, GLPK: {}, C: {}", glpk_cost, c_cost);
        }
        println!("ITER: {}", iter);
    }
    
    println!("Average duration GLPK: {}", average(time_vec[GLPK].as_slice()));
    println!("Average duration Munkres: {}", average(time_vec[RUST].as_slice()));
    println!("Average duration Munkres2: {}", average(time_vec[RUST2].as_slice()));
    println!("Average duration C: {}", average(time_vec[C].as_slice()));
    Ok(())
}

fn sum_up_cost(vect: &Vec<i16>, cost: &[[u16; 1000]; 1000]) -> u32 {
    let mut sum: u32 = 0;
    for (s, d) in vect.iter().enumerate() {
        if *d > -1 // some libraries return -1 for fake assignments
            && cost[s][*d as usize] < BIG_VALUE { // don't sup up fake assignments
            sum += cost[s][*d as usize] as u32;
        }
    }
    return sum;
}

fn read_results_binary(filename: &str, size: usize, cost: &[[u16; 1000]; 1000]) -> (u32, Vec<i16>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut s: usize = 0;
    let mut d: usize = 0;
    let mut cost_sum: u32 = 0;
    let mut ret: Vec<i16> = vec![];

    for line in reader.lines() {
        let flag: usize = line.unwrap().parse().unwrap();
        if flag == 1 {
            ret.push(d as i16);
            if cost[s][d] < BIG_VALUE { // don't sup up fake assignments (non-balanced models)
                cost_sum += cost[s][d] as u32;
            }
        }
        if d == size -1 {
            d = 0;
            s += 1;
        } else { d += 1; }
    }
    return (cost_sum, ret);
}

fn init_cost(cost: &mut [[u16; 1000]; 1000], s_size: usize, d_size: usize) {
    let mut rng = rand::thread_rng();
    let size: usize = cmp::max(d_size, s_size);
    for s in 0 .. size { // supply
        for d in 0 .. size { // demand
            if s < s_size && d < d_size {
                cost[s][d] = rng.gen_range(0..100);
            } else {
                cost[s][d] = BIG_VALUE;
            }
        }
    }
}

fn read_results_index(filename: &str, cost: &[[u16; 1000]; 1000]) -> (u32, Vec<i16>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut s: usize = 0;
    let mut cost_sum: u32 = 0;
    let mut ret: Vec<i16> = vec![];

    for line in reader.lines() {
        let index: i16 = line.unwrap().parse().unwrap();
        ret.push(index as i16);
        if index != -1 {
            cost_sum += cost[s][index as usize] as u32;
        }
        s += 1;
    }
    return (cost_sum, ret);
}

fn write_input_balanced(filename: &str, size: usize, cost: &[[u16; 1000]; 1000]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "{}\n", size).unwrap();
    for s in 0 .. size {
        for d in 0 .. size {
            write!(&mut writer, "{} ", cost[s][d]).unwrap();
        }
        write!(&mut writer, "\n").unwrap();
    }
}

fn write_input(filename: &str, width: usize, length: usize, cost: &[[u16; 1000]; 1000]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "{} ", width).unwrap();
    write!(&mut writer, "{} ", length).unwrap();
    for s in 0 .. width {
        for d in 0 .. length {
            write!(&mut writer, "{} ", cost[s][d]).unwrap();
        }
    }
    writer.flush().unwrap();
}

fn write_input_cpp(filename: &str, width: usize, length: usize, cost: &[[u16; 1000]; 1000]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "{} ", width).unwrap();
    write!(&mut writer, "{} \n", length).unwrap();
    for s in 0 .. width {
        for d in 0 .. length {
            write!(&mut writer, "{} ", cost[s][d]).unwrap();
        }
        write!(&mut writer, "\n").unwrap();
    }
    writer.flush().unwrap();
}

fn munkres(cost: &[[u16; 1000]; 1000], cab_size: usize, order_size: usize) -> Vec<i16> {
    let mut ret: Vec<i16> = vec![];
    let mut matrix: Vec<i32> = vec![];
    
    for s in 0 .. cab_size { // supply
        for d in 0 .. order_size { // demand
            matrix.push(cost[s][d] as i32);
        }
    }
    let assignment = minimize(&matrix, cab_size, order_size);
    
    for s in assignment {
        if s.is_some() {
            ret.push(s.unwrap() as i16);
        } else {
            ret.push(-1);
        }
    }
    return ret;
}

fn munkres2(cost: &[[u16; 1000]; 1000], cab_size: usize, order_size: usize) -> (i32, Vec<usize>) {
    let mut matrix: Vec<Vec<i32>> = vec![];
    
    for s in 0 .. cab_size { // supply
        let mut row: Vec<i32> = vec![];
        for d in 0 .. order_size { // demand
            row.push(cost[s][d] as i32);
        }
        matrix.push(row);
    }
    let weights = Matrix::from_rows(matrix).unwrap();
    return kuhn_munkres_min(&weights);
}

fn average(numbers: &[u64]) -> f32 {
    numbers.iter().sum::<u64>() as f32 / numbers.len() as f32
}
