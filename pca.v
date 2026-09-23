// -*- compile-command: "v run pca.v" -*-
import vsl.la
import vsl.plot as _
import vsl.util
import vtl
import vtl.la as _
import rand
import math

fn main() {
	rand.seed([u32(42), u32(43)])
	x := util.arange(100).map(f32(it))
	vector_scaler_sub :=
		la.vector_add(f32(1), x, f32(-5), []f32{len: x.len, init: 1}) // allocs new arr
	println(vector_scaler_sub)
	y := x.map(2 * it + 3 + rand.f32() * 10) // allocs new arr
	x_tensor := vtl.from_array(x, [x.len, 1])!
	y_tensor := vtl.from_array(y, [y.len, 1])!
	data := vtl.column_stack([x_tensor, y_tensor])! // allocs new arr
	println(data.shape)
	data_t := data.t()! // transpose
	println(data_t.shape)
	split := data.hsplit(2)! // column-wise
	println(split[1].get_nth(0))
	colx := split[0].to_array()
	avgx := la.vector_accum(colx) / colx.len

	coly := split[1].to_array()
	avgy := la.vector_accum(coly) / coly.len

	dcol0 := data.slice([0, data.shape[0]], [0, 1])! // first column
	dcol1 := data.slice([0, data.shape[0]], [1, 2])! // second column

	// res := dcol0 - dcol1 // v compiler error

	centeredx := dcol0.subtract_scalar(f32(avgx))!
	centeredy := dcol1.subtract_scalar(f32(avgy))!
	centered_data := vtl.column_stack([centeredx, centeredy])!
	println(centered_data)

	mut a := la.Matrix.raw(centered_data.shape[0], centered_data.shape[1],
		centered_data.to_array().map(f64(it)))
	mut s := []f64{len: math.min(a.m, a.n)}
	mut u := la.Matrix.new[f64](100, 100)
	mut vt := la.Matrix.new[f64](2, 2)

	la.matrix_svd(mut s, mut u, mut vt, mut a, true)
	println('u: ${u}')
	println('s: ${s}')
	println('vt: ${vt}')

	// mut plt := plot.Plot.new()
	// plt.scatter(x: x, y: y, mode: 'lines+markers')
	// plt.show()!
}
