//! Shared data generation and validation for the v1 and v2 benchmark runners.
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Duration,
};

pub type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    Relu,
    MobileNet,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Predict,
    EndToEnd,
    Reuse,
}

#[derive(Debug)]
pub struct Config {
    pub model: PathBuf,
    pub case: Case,
    pub phase: Phase,
    pub compute: String,
    pub iterations: usize,
    pub warmup: usize,
    pub pool_size: usize,
    pub seed: u32,
    pub output: PathBuf,
}
impl Config {
    pub fn parse() -> BenchResult<Self> {
        let a: Vec<_> = std::env::args().skip(1).collect();
        if a.len() != 9 {
            return Err("usage: RUNNER <model.mlmodelc> <relu|mobilenet> <predict|end-to-end|reuse> <cpu|gpu|ane> <iterations> <warmup> <pool> <seed> <output.csv>".into());
        }
        let case = match a[1].as_str() {
            "relu" => Case::Relu,
            "mobilenet" => Case::MobileNet,
            _ => return Err("unknown model case".into()),
        };
        let phase = match a[2].as_str() {
            "predict" => Phase::Predict,
            "end-to-end" => Phase::EndToEnd,
            "reuse" => Phase::Reuse,
            _ => return Err("unknown phase".into()),
        };
        if !matches!(a[3].as_str(), "cpu" | "gpu" | "ane") {
            return Err("compute must be cpu, gpu, or ane".into());
        }
        let iterations: usize = a[4].parse()?;
        let warmup = a[5].parse()?;
        let pool_size: usize = a[6].parse()?;
        let seed: u32 = a[7].parse()?;
        if !(2..=1024).contains(&pool_size) || iterations < pool_size * 2 || seed == 0 {
            return Err("require seed > 0, pool 2..=1024 and iterations >= 2 * pool".into());
        }
        Ok(Self {
            model: a[0].as_str().into(),
            case,
            phase,
            compute: a[3].clone(),
            iterations,
            warmup,
            pool_size,
            seed,
            output: a[8].as_str().into(),
        })
    }
    pub fn shape(&self) -> &'static [usize] {
        match self.case {
            Case::Relu => &[3],
            Case::MobileNet => &[3, 224, 224],
        }
    }
    pub fn input_name(&self) -> &'static str {
        match self.case {
            Case::Relu => "input",
            Case::MobileNet => "image",
        }
    }
    pub fn output_name(&self) -> &'static str {
        match self.case {
            Case::Relu => "output",
            Case::MobileNet => "classLabelProbs",
        }
    }
    pub fn output_len(&self) -> usize {
        match self.case {
            Case::Relu => 3,
            Case::MobileNet => 1001,
        }
    }
}

fn random(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// A fixed input pool and balanced, deterministically shuffled sample order.
pub struct Data {
    pub inputs: Vec<Vec<f32>>,
    pub order: Vec<usize>,
}
impl Data {
    pub fn new(config: &Config) -> Self {
        let mut seed = config.seed;
        let count: usize = config.shape().iter().product();
        let inputs = (0..config.pool_size)
            .map(|_| {
                (0..count)
                    .map(|_| {
                        let bits = random(&mut seed) >> 8;
                        (bits as f32 / 16_777_215.0) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect();
        let mut order: Vec<usize> = (0..config.iterations)
            .map(|i| i % config.pool_size)
            .collect();
        for i in (1..order.len()).rev() {
            let j = random(&mut seed) as usize % (i + 1);
            order.swap(i, j);
        }
        Self { inputs, order }
    }
}

pub struct Sample {
    pub input: usize,
    pub elapsed: Duration,
    pub prepare: Duration,
    pub call: Duration,
    pub output_copy: Duration,
    pub output: Vec<f32>,
}

/// Verify exact ReLU semantics or a finite, normalized softmax distribution.
pub fn sanity(config: &Config, input: &[f32], output: &[f32]) -> BenchResult<()> {
    if output.len() != config.output_len() {
        return Err(format!(
            "wrong output length: expected {}, got {}",
            config.output_len(),
            output.len()
        )
        .into());
    }
    if output.iter().any(|x| !x.is_finite()) {
        return Err("output contains non-finite values".into());
    }
    match config.case {
        Case::Relu => {
            for (&x, &actual) in input.iter().zip(output) {
                let expected = x.max(0.0);
                if !close(actual, expected) {
                    return Err(format!("ReLU mismatch: got {actual}, expected {expected}").into());
                }
            }
        }
        Case::MobileNet => {
            let sum: f64 = output.iter().map(|&x| x as f64).sum();
            if output.iter().any(|&x| !(-1e-6..=1.000_001).contains(&x))
                || (sum - 1.0).abs() > 0.005
            {
                return Err(format!("invalid MobileNet softmax distribution: sum={sum}").into());
            }
        }
    }
    Ok(())
}
pub fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1e-5 + 1e-3 * a.abs().max(b.abs())
}

/// Validate all observations outside the measured sections, then write raw data.
pub fn finish(config: &Config, data: &Data, samples: &[Sample]) -> BenchResult<()> {
    if samples.len() != config.iterations {
        return Err(format!(
            "expected {} samples, got {}",
            config.iterations,
            samples.len()
        )
        .into());
    }
    let mut references: Vec<Option<&[f32]>> = vec![None; config.pool_size];
    for sample in samples {
        sanity(config, &data.inputs[sample.input], &sample.output)?;
        if let Some(reference) = references[sample.input] {
            if sample
                .output
                .iter()
                .zip(reference)
                .any(|(&a, &b)| !close(a, b))
            {
                return Err(format!("output unstable for input {}", sample.input).into());
            }
        } else {
            references[sample.input] = Some(&sample.output);
        }
    }
    if references.iter().any(Option::is_none) {
        return Err("not every input was exercised".into());
    }
    if config.case == Case::MobileNet {
        let first = references[0].expect("all inputs were exercised");
        let varies = references
            .iter()
            .flatten()
            .any(|other| other.iter().zip(first).any(|(&a, &b)| !close(a, b)));
        if !varies {
            return Err("MobileNet outputs did not vary across the randomized input pool".into());
        }
    }
    let mut file = BufWriter::new(File::create(&config.output)?);
    writeln!(
        file,
        "sample,input_id,elapsed_ns,prepare_ns,call_ns,output_copy_ns,output"
    )?;
    for (i, sample) in samples.iter().enumerate() {
        write!(
            file,
            "{i},{},{},{},{},{},",
            sample.input,
            sample.elapsed.as_nanos(),
            sample.prepare.as_nanos(),
            sample.call.as_nanos(),
            sample.output_copy.as_nanos()
        )?;
        for (j, value) in sample.output.iter().enumerate() {
            if j > 0 {
                write!(file, ";")?;
            }
            write!(file, "{value:.9e}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(case: Case) -> Config {
        Config {
            model: "unused".into(),
            case,
            phase: Phase::Predict,
            compute: "cpu".into(),
            iterations: 16,
            warmup: 2,
            pool_size: 4,
            seed: 42,
            output: "unused".into(),
        }
    }

    #[test]
    fn data_is_reproducible_balanced_and_varied() {
        let a = Data::new(&config(Case::Relu));
        let b = Data::new(&config(Case::Relu));
        assert_eq!(a.inputs, b.inputs);
        assert_eq!(a.order, b.order);
        assert_ne!(a.inputs[0], a.inputs[1]);
        for id in 0..4 {
            assert_eq!(a.order.iter().filter(|&&n| n == id).count(), 4);
        }
        let correct: Vec<_> = a.inputs[0].iter().map(|v| v.max(0.0)).collect();
        assert!(sanity(&config(Case::Relu), &a.inputs[0], &correct).is_ok());
        assert!(sanity(&config(Case::Relu), &a.inputs[0], &[f32::NAN, 0.0, 0.0]).is_err());
    }

    #[test]
    fn rejects_constant_mobilenet_outputs() {
        let config = config(Case::MobileNet);
        let data = Data::new(&config);
        let mut output = vec![0.0; 1001];
        output[0] = 1.0;
        let samples: Vec<_> = data
            .order
            .iter()
            .map(|&input| Sample {
                input,
                elapsed: Duration::ZERO,
                prepare: Duration::ZERO,
                call: Duration::ZERO,
                output_copy: Duration::ZERO,
                output: output.clone(),
            })
            .collect();
        assert!(finish(&config, &data, &samples).is_err());
    }

    #[test]
    fn rejects_unnormalized_probabilities() {
        let output = vec![0.0; 1001];
        assert!(sanity(&config(Case::MobileNet), &[], &output).is_err());
    }
}
