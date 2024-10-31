use std::cmp;
use std::collections::HashSet;
use std::hash::Hash;
use std::fs::{File, remove_file};
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
use std::time::Instant;
use chrono::Utc;
use rand::Rng;
use hungarian::minimize;
use pathfinding::prelude::{kuhn_munkres_min, Matrix};
use strum::{EnumIter, IntoEnumIterator, IntoStaticStr};

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
const BIG_VALUE: u16 = 65255;
const MIN_VALUE: u16 = 30;
const MAX_VALUE: u16 = 1800;
const MAX_ITER: usize = 1;

#[derive(PartialEq, Eq, Clone, Debug, EnumIter, IntoStaticStr)]
enum Solvers {
    GLPK,
    RUST,
    RUST2,
    C,
    CPP,
    CPP2,
    PYTHON,
    C2,
    CPP3,
    PYTHON2,
    CPP4,
    LCM
}

const DSIZE: usize = 2001;
const SSIZE: usize = 2001;

fn main()  -> std::io::Result<()> {
    let mut cost: [[u16; DSIZE]; SSIZE] = [[0; DSIZE]; SSIZE];
    let demand_size: usize = 1000;
    let supply_size: usize = 2000;
    let mut cost_vec: [Vec<u32>; 15] = [const { Vec::new() }; 15];
    let mut time_vec: [Vec<u128>; 15] = [const { Vec::new() }; 15];

    init_cost(&mut cost);

    for iter in 0 .. MAX_ITER {
        println!("Iter {} start: {:?}", iter, Utc::now());
        random_cost(&mut cost, supply_size, demand_size);
        
        // ----------------- RUST ----------------------
        // https://crates.io/crates/hungarian
        let munk_cost = run_munkres(demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // --------------- GLPK ------------------
        //run("python3 glpk.py", Solvers::GLPK, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // ------------------ RUST faster ------------------
        // https://crates.io/crates/pathfinding/4.3.1
        // !! "number of rows must not be larger than number of columns"
        // then 500*8000 needs 8000x8000
        let munk_cost = run_munkres2(demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // ---------------- C ------------------------
        // https://github.com/xg590/munkres
        // fails e.g. with n=1000
        // Segm fault e.g. with n=1000
        //run("./munkres1", Solvers::C, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // ---------------- C++ ------------------------
        // https://github.com/mcximing/hungarian-algorithm-cpp
        // 1000x2000, 30..1800: plan is invalid
        // 500x8000, 0..30: duplicates found
        //run("./munkres2", Solvers::CPP, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // ---------------- C++ ------------------------
        // https://github.com/phoemur/hungarian_algorithm/blob/master/hungarian.cpp
        // SLOW, 500x8000: very slow
        run("./munkres3", Solvers::CPP2, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);
        
        // ---------------- Python
        // https://software.clapper.org/munkres/
        // python3 -m pip install munkres
        // SLOW
        generate_python("munk.py", supply_size, demand_size, &cost);
        run("python3 munk.py", Solvers::PYTHON, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec); 

        // ---------------- Python LAPJV
        // https://github.com/src-d/lapjv
        // python3 -m pip install lapjv
        // 1000x2000: ValueError: "cost_matrix" must be a square 2D numpy array
        //generate_python2("munk2.py", supply_size, demand_size, &cost);
        //run("python3 munk2.py", Solvers::PYTHON2, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);
        
        // ---------- C ----
        // https://ranger.uta.edu/~weems/NOTES5311/hungarian.c
        // hangs when non-balanced, at least 1000x2000, 30..1800
        //run("./munkres4", Solvers::C2, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);
        // !! no use to read as it hang when non-balance

        // ---------------- C++ -----------------
        // https://github.com/yongyanghz/LAPJV-algorithm-c
        // this implementation assumes quadratic cost matrix, balanced models
        //run("./lap1", Solvers::CPP3, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);
     
        // https://github.com/aaron-michaux/munkres-algorithm.git
        // does not compile on Mac
        // SLOW on Ubuntu
        // 1000x2000, 30..1800: non-optimal value + slow
        run("./munkres6", Solvers::CPP4, munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);

        // Low Cost Method, just for comparison
        run_lcm(munk_cost, demand_size, supply_size, &cost, &mut cost_vec, &mut time_vec);
    }
    
    for solv in Solvers::iter() {
        let name: &'static str = solv.clone().into();
        if time_vec[solv.clone() as usize].len() == 0 {
            println!("{}: no data", name);
        } else {
            println!("{}: Avg: {}, Min: {}, Max: {}", name, average(time_vec[solv.clone() as usize].as_slice()), 
                time_vec[solv.clone() as usize].iter().min().unwrap(), time_vec[solv as usize].iter().max().unwrap());
        }
    }
    Ok(())
}

fn run_lcm(exp_val: u32, d_size: usize, s_size: usize, cost: &[[u16; DSIZE]; SSIZE], cost_vec: &mut [Vec<u32>; 15], time_vec: &mut [Vec<u128>; 15]) {
    let min_size: usize = cmp::min(d_size, s_size);
    let start = Instant::now();
    let (lcm_cost, ret) = lcm(&cost, s_size, d_size);
    time_vec[Solvers::LCM as usize].push(start.elapsed().as_millis());
    if ret.len() != min_size || !no_duplicates(ret) {
        println!("LCM: plan is invalid");
    }
    // --------------- COMPARING RESULTS
    if exp_val != lcm_cost {
        println!("LCM is worse rust: {} lcm: {}", exp_val, lcm_cost);
    }
    cost_vec[Solvers::LCM as usize].push(lcm_cost);
}

fn run_munkres(d_size: usize, s_size: usize, cost: &[[u16; DSIZE]; SSIZE], cost_vec: &mut [Vec<u32>; 15], time_vec: &mut [Vec<u128>; 15]) -> u32 {
    let min_size: usize = cmp::min(d_size, s_size);
    let start = Instant::now();
    let munk = munkres(&cost, s_size, d_size);
    
    time_vec[Solvers::RUST as usize].push(start.elapsed().as_millis());
    let munk_cost: u32 = sum_up_cost(&munk, &cost);

    // QA
    let (_, values) = rm_minusone(&munk);
   /* if supply_size - demand_size != minus_count as usize {
        println!("Minus count is different than size diff: count: {}, demand: {}, supply: {}",
            minus_count, demand_size, supply_size);
    }
    */
    if values.len() != min_size || !no_duplicates(values) {
        println!("Rust: plan is invalid");
    }
    cost_vec[Solvers::RUST as usize].push(munk_cost);
    //println!("Munkres ({}): {:?}", munk_cost, munk);
    return munk_cost;
}

fn run_munkres2(d_size: usize, s_size: usize, cost: &[[u16; DSIZE]; SSIZE], cost_vec: &mut [Vec<u32>; 15], time_vec: &mut [Vec<u128>; 15]) -> u32 {
    let max_size: usize = cmp::max(d_size, s_size);
    let start = Instant::now();
    let (_, ret) = munkres2(&cost, max_size, max_size);
    time_vec[Solvers::RUST2 as usize].push(start.elapsed().as_millis());
    
    let mut munk2_cost = 0;
    for (s, d) in ret.iter().enumerate() {
        if cost[s][*d as usize] < BIG_VALUE {
            munk2_cost += cost[s][*d as usize] as u32;
        }
    }
    cost_vec[Solvers::RUST2 as usize].push(munk2_cost);
    
    //println!("Munkres2 ({}): {:?}", munk2_cost, ret);
    return munk2_cost;
}

fn run(cmd: &str, key: Solvers, exp_val: u32, d_size: usize, s_size: usize, cost: &[[u16; DSIZE]; SSIZE],
       cost_vec: &mut [Vec<u32>; 15], time_vec: &mut [Vec<u128>; 15]) {
    println!("{}...", cmd);
    let max_size: usize = cmp::max(d_size, s_size);
    let min_size: usize = cmp::min(d_size, s_size);
    if key == Solvers::GLPK {
        write_input_balanced("input.txt", max_size, &cost);
    } else {
        write_input("input.txt", s_size, d_size, &cost);
    }
    match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
    
    Command::new("sh").arg("-c").arg(cmd).output().expect("failed to execute process");
    
    let (elapsed, cost, ret) =  match key {
        Solvers::GLPK => read_results_binary("output.txt", max_size, &cost),
        Solvers::CPP2 => read_square_matrix("output.txt", &cost),
        Solvers::PYTHON => read_python_row_col("output.txt", &cost),
                        _  => read_results_index("output.txt", &cost),
                    };
    time_vec[key.clone() as usize].push(elapsed);
    cost_vec[key as usize].push(cost);
    if ret.len() != min_size && ret.len() != max_size {
        println!("Plan is invalid, expected size: {}, returned number of rows: {}", min_size, ret.len());
    }
    if !no_duplicates(ret) {
        println!("Plan is invalid, duplicated found");
    }
    if exp_val != cost {
        println!("{}: expected value {} != {}", cmd, exp_val, cost);
    }
}

fn  rm_minusone(vec: &Vec<i16>) -> (i16, Vec<i16>) {
    let mut ret: Vec<i16> = vec![];
    let mut counter: i16 = 0;
    for v in vec.iter() {
        if *v != -1 {
            ret.push(*v);
        } else {
            counter += 1;
        }
    }
    return (counter, ret);
}

fn generate_python(filename: &str, width: usize, length: usize, cost: &[[u16; DSIZE]; SSIZE]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "import datetime\nfrom munkres import Munkres\nmatrix = [").unwrap();
    for s in 0 .. width {
        write!(&mut writer, "[").unwrap();
        for d in 0 .. length {
            write!(&mut writer, "{}", cost[s][d]).unwrap();
            if d < length -1 {
                write!(&mut writer, ",").unwrap();
            }
        }
        write!(&mut writer, "]").unwrap();
        if s < width - 1 {
            write!(&mut writer, ",").unwrap();
        }
        write!(&mut writer, "\n").unwrap();
    }
    write!(&mut writer, "]\na = datetime.datetime.now()\nm = Munkres()\nindexes = m.compute(matrix)\n").unwrap();
    write!(&mut writer, "b = datetime.datetime.now()\nc = b - a\nmillis = int(c.total_seconds() * 1000)\n").unwrap();
    write!(&mut writer, "f = open(\"output.txt\", \"w\")\nf.write (\"%d\\n\" % (millis))\n").unwrap();
    write!(&mut writer, "for row, column in indexes:\n\tf.write (\"%d %d\\n\" % (row, column))\n").unwrap(); 
    write!(&mut writer, "f.close()\n").unwrap(); 
}

fn generate_python2(filename: &str, width: usize, length: usize, cost: &[[u16; DSIZE]; SSIZE]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "import datetime\nfrom lapjv import lapjv\nmatrix = [").unwrap();
    for s in 0 .. width {
        write!(&mut writer, "[").unwrap();
        for d in 0 .. length {
            write!(&mut writer, "{}", cost[s][d]).unwrap();
            if d < length -1 {
                write!(&mut writer, ",").unwrap();
            }
        }
        write!(&mut writer, "]").unwrap();
        if s < width - 1 {
            write!(&mut writer, ",").unwrap();
        }
        write!(&mut writer, "\n").unwrap();
    }
    write!(&mut writer, "]\na = datetime.datetime.now()\nrow, col, _ = lapjv(matrix)\n").unwrap();
    write!(&mut writer, "b = datetime.datetime.now()\nc = b - a\nmillis = int(c.total_seconds() * 1000)\n").unwrap();
    write!(&mut writer, "f = open(\"output.txt\", \"w\")\nf.write (\"%d\\n\" % (millis))\n").unwrap();
    write!(&mut writer, "for r in row:\n\tf.write (\"%d\\n\" % (r))\n").unwrap(); 
    write!(&mut writer, "f.close()\n").unwrap(); 
}

fn sum_up_cost(vect: &Vec<i16>, cost: &[[u16; DSIZE]; SSIZE]) -> u32 {
    let mut sum: u32 = 0;
    for (s, d) in vect.iter().enumerate() {
        if *d > -1 // some libraries return -1 for fake assignments
            && cost[s][*d as usize] < BIG_VALUE { // don't sup up fake assignments
            sum += cost[s][*d as usize] as u32;
        }
    }
    return sum;
}

fn read_square_matrix(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u128, u32, Vec<i16>) {
    let mut ret: Vec<i16> = vec![];
    let mut sum: u32 = 0;
    let f = BufReader::new(File::open(filename).unwrap());
    let mut first_line = true; // elapsed time in the first line
    let mut elapsed: u128 = 0;

    for (i, line) in f.lines().enumerate() {
        if first_line {
            first_line = false;
            elapsed = line.unwrap().parse().unwrap();
            continue;
        }
        for (j, number) in line.unwrap().split(char::is_whitespace).enumerate() {
            let flag = match number.trim().parse::<i8>() {
                Ok(t) => t,
                Err(_e) => -1,
            };
            if flag == -1 { // the first whitespace
                continue;
            }
            if flag == 1 {
                let idx = j -2; // two whitespaces to be skipped
                ret.push(j as i16);
                if cost[i-1][idx] < BIG_VALUE { // -1 as the first line contains elapsed time
                    sum += cost[i-1][idx] as u32;
                }
                // we could break here from the inner loop
            }
        }
    }
    return (elapsed, sum, ret);
}

fn read_python_row_col(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u128, u32, Vec<i16>) {
    let mut ret: Vec<i16> = vec![];
    let mut sum: u32 = 0;
    let mut row: usize = 0;
    let mut col: usize;
    let mut first_line = true; // elapsed time in the first line
    let mut elapsed: u128 = 0;
    let f = BufReader::new(File::open(filename).unwrap());

    for (_i, line) in f.lines().enumerate() {
        if first_line {
            first_line = false;
            elapsed = line.unwrap().parse().unwrap();
            continue;
        }
        for (j, number) in line.unwrap().split(char::is_whitespace).enumerate() {
            if j == 0 {
                row = number.trim().parse::<usize>().unwrap();
            } else {
                col = number.trim().parse::<usize>().unwrap();
                ret.push(col as i16);
                if cost[row][col] < BIG_VALUE {
                    sum += cost[row][col] as u32;
                }
            }
        }
    }
    return (elapsed, sum, ret);
}

fn read_results_binary(filename: &str, size: usize, cost: &[[u16; DSIZE]; SSIZE]) -> (u128, u32, Vec<i16>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut s: usize = 0;
    let mut d: usize = 0;
    let mut cost_sum: u32 = 0;
    let mut ret: Vec<i16> = vec![];
    let mut first_line = true; // elapsed time in the first line
    let mut elapsed: u128 = 0;
    for line in reader.lines() {
        if first_line {
            first_line = false;
            elapsed = line.unwrap().parse().unwrap();
            continue;
        }
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
    return (elapsed, cost_sum, ret);
}

fn init_cost(cost: &mut [[u16; DSIZE]; SSIZE]) {
    for s in 0 .. SSIZE { // supply
        for d in 0 .. DSIZE { // demand
            cost[s][d] = BIG_VALUE;
        }
    }
}

fn random_cost(cost: &mut [[u16; DSIZE]; SSIZE], s_size: usize, d_size: usize) {
    let mut rng = rand::thread_rng();
    for s in 0 .. s_size { // supply
        for d in 0 .. d_size { // demand
            cost[s][d] = rng.gen_range(MIN_VALUE..MAX_VALUE);
        }
    }
}

fn read_results_index(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u128, u32, Vec<i16>) {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let mut s: usize = 0;
    let mut cost_sum: u32 = 0;
    let mut ret: Vec<i16> = vec![];
    let mut first_line = true; // elapsed time in the first line
    let mut elapsed: u128 = 0;
    for line in reader.lines() {
        if first_line {
            first_line = false;
            elapsed = line.unwrap().parse().unwrap();
            continue;
        }
        let index: i16 = line.unwrap().parse().unwrap();
        ret.push(index as i16);
        if index != -1 && index < DSIZE as i16 // such index means a fake customer in order to get the square matrix
            && cost[s][index as usize] < BIG_VALUE {
            cost_sum += cost[s][index as usize] as u32;
        }
        s += 1;
    }
    return (elapsed, cost_sum, ret);
}

fn write_input_balanced(filename: &str, size: usize, cost: &[[u16; DSIZE]; SSIZE]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "{}\n", size).unwrap();
    for s in 0 .. size {
        for d in 0 .. size {
            write!(&mut writer, "{} ", cost[s][d]).unwrap();
        }
        write!(&mut writer, "\n").unwrap();
    }
}

fn write_input(filename: &str, width: usize, length: usize, cost: &[[u16; DSIZE]; SSIZE]) {
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

/*
fn write_input_cpp(filename: &str, width: usize, length: usize, cost: &[[u16; DSIZE]; SSIZE]) {
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
*/

fn munkres(cost: &[[u16; DSIZE]; SSIZE], cab_size: usize, order_size: usize) -> Vec<i16> {
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

fn munkres2(cost: &[[u16; DSIZE]; SSIZE], cab_size: usize, order_size: usize) -> (i32, Vec<usize>) {
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

fn average(numbers: &[u128]) -> f32 {
    return if numbers.len() == 0 {
        -1.0
    } else {
        numbers.iter().sum::<u128>() as f32 / numbers.len() as f32
    };
}

fn no_duplicates<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Hash + Eq,
{
    let mut set = HashSet::new();
    iter.into_iter().all(move |x| set.insert(x))
}

fn lcm(cost: &[[u16; DSIZE]; SSIZE], cab_size: usize, order_size: usize) -> (u32, Vec<usize>) {
    let mut cabs: [bool; SSIZE] = [false; SSIZE];
    let mut orders: [bool; DSIZE] = [false; DSIZE];
    let mut lcm_min_val;
    let mut pairs: Vec<usize> = vec![];
    let mut sum_cost: u32 = 0;
    for _ in 0..cmp::min(cab_size, order_size) { // we need to repeat the search (cut off rows/columns) 'howMany' times
        lcm_min_val = BIG_VALUE;
        let mut smin: usize = SSIZE;
        let mut dmin: usize = DSIZE;
        // now find the minimal element in the whole matrix
        let mut found = false;
        for cab in 0..cab_size {
            if cabs[cab] == true {
                continue;
            }
            for order in 0..order_size {
                if orders[order] == false && cost[cab][order] < lcm_min_val {
                    lcm_min_val = cost[cab][order];
                    smin = cab;
                    dmin = order;
                    if lcm_min_val == MIN_VALUE { // you can't have a better solution
                        found = true;
                        break;
                    }
                }
            }
            if found {
                break; // yes, we could have loop labels and break two of them here, but this is for migration to C
            }
        }
        if lcm_min_val == BIG_VALUE {
            println!("LCM minimal cost is big_cost - no more interesting stuff here");
            break;
        }
        // binding cab to the customer order
        pairs.push(dmin);
        sum_cost += cost[smin][dmin] as u32;
        // removing the "columns" and "rows" from a virtual matrix
        cabs[smin] = true;
        orders[dmin] = true;
    }
    return (sum_cost, pairs);
}
