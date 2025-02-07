use coreml_rs::mlmodel::{CoreMLModel, Store};
use ndarray::Array4;

pub fn main() {
    let s = Store::new();
    let mut m = model_load_and_compile(&s);
    let time = std::time::Instant::now();
    let idx = s.insert(Array4::<f32>::zeros((1, 3, 512, 512)));
    m.add_input_f32("img", s.bind_ref(idx));
    let p = time.elapsed();
    println!("create and add input {} ms", p.as_millis());
    let time = std::time::Instant::now();
    let output = m.predict();
    let p = time.elapsed();
    println!("predict {} ms", p.as_millis());
    // let v = output.unwrap().outputF32("add".to_string());
    // let d = time.elapsed() - p;
    // println!("copy output {} ms", d.as_millis());
    // let _output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();
    // let d = time.elapsed() - (p + d);
    // println!("transmute to shape {} ms", d.as_millis());
}

fn model_load_and_compile<'a>(s: &'a Store) -> CoreMLModel<'a> {
    let mut model = CoreMLModel::new("./demo/test.mlpackage", s);
    model.compile();
    model.load();
    // println!("model description:\n{:#?}", model.description());
    return model;
}
