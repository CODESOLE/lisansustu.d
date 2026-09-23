// -*- compile-command: "v run game.v" -*-
import rand
import vsl.la

fn main() {
	n, p := 1000_000, 0.5
	mut played_round := []int{len: n}
	for i in 0 .. n {
		mut nloss := 0
		mut nround := 0
		for nloss < 2 {
			nround++
			if rand.f64() < p {
				nloss = 0
			} else {
				nloss++
			}
		}
		played_round[i] = nround
	}
	sum := la.vector_accum(played_round)
	println(f64(sum) / f64(n))
}
