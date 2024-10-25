use std::cmp;
use std::collections::HashSet;
use std::hash::Hash;
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
const BIG_VALUE: u16 = 65255;
const MAX_ITER: usize = 5;

const GLPK:usize = 0;
const RUST:usize = 1;
const RUST2:usize = 2;
//const C:usize = 3;
const CPP:usize = 4;
const CPP2:usize = 5;
const PYTHON:usize = 6;
const C2:usize = 7;
const CPP3:usize = 8;
const PYTHON2:usize = 9;
const CPP4:usize = 10;

const DSIZE: usize = 2001;
const SSIZE: usize = 2001;

fn main()  -> std::io::Result<()> {
    let mut cost: [[u16; DSIZE]; SSIZE] = [[0; DSIZE]; SSIZE];
    let demand_size: usize = 1000;
    let supply_size: usize = 1000;
    let max_size: usize = cmp::max(demand_size, supply_size);
    let mut cost_vec: [Vec<u32>; 15] = [const { Vec::new() }; 15];
    let mut time_vec: [Vec<u128>; 15] = [const { Vec::new() }; 15];

    init_cost(&mut cost);

    for iter in 0 .. MAX_ITER {
        random_cost(&mut cost, supply_size, demand_size);
        
        // --------------- GLPK ------------------
        write_input_balanced("input.txt", max_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        let mut start = Instant::now();
    /*    Command::new("sh")
                .arg("-c").arg("python3 glpk.py").output().expect("failed to execute process");
        time_vec[GLPK].push(start.elapsed().as_millis());

        let (glpk_cost, glpk) = read_results_binary("output.txt", max_size, &cost);
        cost_vec[GLPK].push(glpk_cost);
        //println!("GLPK ({}): {:?}", glpk_cost, glpk);
        //println!("GLPK cost: {}", glpk_cost);
      */  
        // ----------------- RUST ----------------------
        // https://crates.io/crates/hungarian
        start = Instant::now();
        let munk = munkres(&cost, supply_size, demand_size);
        
        time_vec[RUST].push(start.elapsed().as_millis());
        let munk_cost: u32 = sum_up_cost(&munk, &cost);

        // QA
        let (minus_count, values) = rm_minusone(&munk);
       /* if supply_size - demand_size != minus_count as usize {
            println!("Minus count is different than size diff: count: {}, demand: {}, supply: {}",
                minus_count, demand_size, supply_size);
        }
        */
        if (!no_duplicates(values)) {
            println!("Rust: plan has duplicates");
        }
        cost_vec[RUST].push(munk_cost);
        //println!("Munkres ({}): {:?}", munk_cost, munk);

        // ------------------ RUST faster ------------------
        // https://crates.io/crates/pathfinding/4.3.1
        // !! "number of rows must not be larger than number of columns"
       /*
        start = Instant::now();
        let (_, ret) = munkres2(&cost, max_size, max_size);
        time_vec[RUST2].push(start.elapsed().as_millis());
        
        let mut munk2_cost = 0;
        for (s, d) in ret.iter().enumerate() {
            if cost[s][*d as usize] < BIG_VALUE {
                munk2_cost += cost[s][*d as usize] as u32;
            }
        }
        cost_vec[RUST2].push(munk2_cost);
        
        //println!("Munkres2 ({}): {:?}", munk2_cost, ret);
        */

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
        time_vec[CPP].push(start.elapsed().as_millis());
        let (c_cost, c_ret) = read_results_index("output.txt", &cost);
        //println!("CPP ret ({}) {:?}", c_cost, c_ret);
        cost_vec[CPP].push(c_cost);

        // ---------------- C++ ------------------------
        // https://github.com/phoemur/hungarian_algorithm/blob/master/hungarian.cpp
    /*    write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("./munkres3").output().expect("failed to execute process");
        time_vec[CPP2].push(start.elapsed().as_millis());
        let (c_cost2, c_ret) = read_square_matrix("output.txt", &cost);
        //println!("CPP2 ret ({}) {:?}", c_cost2, c_ret);
        cost_vec[CPP2].push(c_cost2);
        
        // ---------------- Python
        // https://software.clapper.org/munkres/
        // python3 -m pip install munkres
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        generate_python("munk.py", supply_size, demand_size, &cost);
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("python3 munk.py").output().expect("failed to execute process");
        time_vec[PYTHON].push(start.elapsed().as_millis());
        let (python_cost, c_ret) = read_python_row_col("output.txt", &cost);
        //println!("Python ret ({}) {:?}", python_cost, c_ret);
        cost_vec[PYTHON].push(python_cost);        
      */  

       // ---------------- Python LAPJV
        // https://github.com/src-d/lapjv
        // python3 -m pip install lapjv
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        generate_python2("munk2.py", supply_size, demand_size, &cost);
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("python3 munk2.py").output().expect("failed to execute process");
        time_vec[PYTHON2].push(start.elapsed().as_millis());
        let (python2_cost, c_ret) = read_results_index("output.txt", &cost);
        //println!("Python ret ({}) {:?}", python_cost, c_ret);
        cost_vec[PYTHON2].push(python2_cost);        

        // ---------- C ----
        // https://ranger.uta.edu/~weems/NOTES5311/hungarian.c
        // hangs when non-balanced
        write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh")
                .arg("-c").arg("./munkres4").output().expect("failed to execute process");
        time_vec[C2].push(start.elapsed().as_millis());
        let (c2_cost, c_ret) = read_results_index("output.txt", &cost);
        cost_vec[C2].push(c2_cost);

        // !! no use to read as it hang when non-balanced

        // ---------------- C++ -----------------
        // https://github.com/yongyanghz/LAPJV-algorithm-c
        // this implementation assumes quadratic cost matrix, balanced models
     /*  write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh").arg("-c").arg("./lap1").output().expect("failed to execute process");
        time_vec[CPP3].push(start.elapsed().as_millis());
        let (lap_cost, c_ret) = read_results_index("output.txt", &cost);
        //println!("CPP ret ({}) {:?}", c_cost, c_ret);
        cost_vec[CPP3].push(lap_cost);
        */

        // https://github.com/aaron-michaux/munkres-algorithm.git
        write_input("input.txt", supply_size, demand_size, &cost);
        match remove_file("output.txt") { Ok(_) => {} Err(_) => {} };
        start = Instant::now();
        Command::new("sh").arg("-c").arg("./munkres6").output().expect("failed to execute process");
        time_vec[CPP4].push(start.elapsed().as_millis());
        let (cpp4_cost, c_ret) = read_results_index("output.txt", &cost);
        cost_vec[CPP4].push(cpp4_cost);


        // --------------- COMPARING RESULTS
        /*
        if glpk_cost != munk_cost {
            println!("Cost does not equal, GLPK: {}, Munkres: {}", glpk_cost, munk_cost);
        }
        if glpk_cost != munk2_cost as u32 {
            println!("Cost does not equal, GLPK: {}, Munkres2: {}", glpk_cost, munk2_cost);
        }
        if glpk_cost != c_cost as u32 {
            println!("Cost does not equal, GLPK: {}, C++: {}", glpk_cost, c_cost);
        }
        if glpk_cost != c_cost2 as u32 {
            println!("Cost does not equal, GLPK: {}, C++2: {}", glpk_cost, c_cost2);
        }
        if glpk_cost != python_cost as u32 {
            println!("Cost does not equal, GLPK: {}, Python: {}", glpk_cost, python_cost);
        }
        if glpk_cost != lap_cost as u32 {
            println!("Cost does not equal, GLPK: {}, LAP: {}", glpk_cost, lap_cost);
        }
        */
        if (munk_cost != c_cost || munk_cost != c2_cost || munk_cost != python2_cost || munk_cost != cpp4_cost) {
             println!("Costs do not equal: rust: {} cpp: {} c2: {}", munk_cost, c_cost, c2_cost);
        }
        println!("ITER: {}", iter);
    }
    
    //println!("GLPK: Avg: {}, Min: {}, Max: {}", average(time_vec[GLPK].as_slice()), time_vec[GLPK].iter().min().unwrap(), time_vec[GLPK].iter().max().unwrap());
    println!("Munkres: Avg: {}, Min: {}, Max: {}", average(time_vec[RUST].as_slice()), time_vec[RUST].iter().min().unwrap(), time_vec[RUST].iter().max().unwrap());
    //println!("Munkres2: Avg: {}, Min: {}, Max: {}", average(time_vec[RUST2].as_slice()), time_vec[RUST2].iter().min().unwrap(), time_vec[RUST2].iter().max().unwrap());
    println!("CPP: Avg: {}, Min: {}, Max: {}", average(time_vec[CPP].as_slice()), time_vec[CPP].iter().min().unwrap(), time_vec[CPP].iter().max().unwrap());
    //println!("CPP2: Avg: {}, Min: {}, Max: {}", average(time_vec[CPP2].as_slice()), time_vec[CPP2].iter().min().unwrap(), time_vec[CPP2].iter().max().unwrap());
    //println!("Python: Avg: {}, Min: {}, Max: {}", average(time_vec[PYTHON].as_slice()), time_vec[PYTHON].iter().min().unwrap(), time_vec[PYTHON].iter().max().unwrap());
    println!("C2: Avg: {}, Min: {}, Max: {}", average(time_vec[C2].as_slice()), time_vec[C2].iter().min().unwrap(), time_vec[C2].iter().max().unwrap());
    println!("Python2: Avg: {}, Min: {}, Max: {}", average(time_vec[PYTHON2].as_slice()), time_vec[PYTHON2].iter().min().unwrap(), time_vec[PYTHON2].iter().max().unwrap());
    //println!("LAP: Avg: {}, Min: {}, Max: {}", average(time_vec[CPP3].as_slice()), time_vec[CPP3].iter().min().unwrap(), time_vec[CPP3].iter().max().unwrap());
    println!("CPP4: Avg: {}, Min: {}, Max: {}", average(time_vec[CPP4].as_slice()), time_vec[CPP4].iter().min().unwrap(), time_vec[CPP4].iter().max().unwrap());
    Ok(())
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
    write!(&mut writer, "from munkres import Munkres\nmatrix = [").unwrap();
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
    write!(&mut writer, "]\nm = Munkres()\nindexes = m.compute(matrix)\nf = open(\"output.txt\", \"w\")\n").unwrap();
    write!(&mut writer, "for row, column in indexes:\n\tf.write (\"%d %d\\n\" % (row, column))\n").unwrap(); 
    write!(&mut writer, "f.close()\n").unwrap(); 
}

fn generate_python2(filename: &str, width: usize, length: usize, cost: &[[u16; DSIZE]; SSIZE]) {
    let mut writer = File::create(filename).expect("creation failed");
    write!(&mut writer, "from lapjv import lapjv\nmatrix = [").unwrap();
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
    write!(&mut writer, "]\nrow, col, _ = lapjv(matrix)\nf = open(\"output.txt\", \"w\")\n").unwrap();
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

fn read_square_matrix(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u32, Vec<i16>) {
    let mut ret: Vec<i16> = vec![];
    let mut sum: u32 = 0;
    let f = BufReader::new(File::open(filename).unwrap());

    for (i, line) in f.lines().enumerate() {
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
                if cost[i][idx] < BIG_VALUE {
                    sum += cost[i][idx] as u32;
                }
                // we could break here from the inner loop
            }
        }
    }
    return (sum, ret);
}

fn read_python_row_col(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u32, Vec<i16>) {
    let mut ret: Vec<i16> = vec![];
    let mut sum: u32 = 0;
    let mut row: usize = 0;
    let mut col: usize;
    let f = BufReader::new(File::open(filename).unwrap());

    for (_i, line) in f.lines().enumerate() {
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
    return (sum, ret);
}

fn read_results_binary(filename: &str, size: usize, cost: &[[u16; DSIZE]; SSIZE]) -> (u32, Vec<i16>) {
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
            cost[s][d] = rng.gen_range(0..30);
        }
    }
}

fn read_results_index(filename: &str, cost: &[[u16; DSIZE]; SSIZE]) -> (u32, Vec<i16>) {
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
    numbers.iter().sum::<u128>() as f32 / numbers.len() as f32
}

fn no_duplicates<T>(iter: T) -> bool
where
    T: IntoIterator,
    T::Item: Hash + Eq,
{
    let mut set = HashSet::new();
    iter.into_iter().all(move |x| set.insert(x))
}
