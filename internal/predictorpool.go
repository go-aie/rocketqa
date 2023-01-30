package internal

import (
	"context"
	"runtime"
	"sync/atomic"

	"github.com/jackc/puddle/v2"
	paddle "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

// PredictorPool is a predictor pool for concurrent inferences.
//
// See also:
// - https://github.com/PaddlePaddle/Paddle/issues/17288
// - https://www.paddlepaddle.org.cn/inference/master/guides/performance_tuning/multi_thread.html
type PredictorPool struct {
	pool *puddle.Pool[*paddle.Predictor]
}

func NewPredictorPool(config *paddle.Config, size int) *PredictorPool {
	if size < 1 {
		size = runtime.NumCPU()
	}

	var first int64
	mainPredictor := paddle.NewPredictor(config)

	pool, err := puddle.NewPool(&puddle.Config[*paddle.Predictor]{
		Constructor: func(context.Context) (*paddle.Predictor, error) {
			if atomic.CompareAndSwapInt64(&first, 0, 1) {
				return mainPredictor, nil
			} else {
				return mainPredictor.Clone(), nil
			}
		},
		Destructor: func(value *paddle.Predictor) {},
		MaxSize:    int32(size),
	})
	if err != nil {
		panic(err)
	}

	// Pre-create all predictors since they can not be created on demand.
	//
	// The root cause is that predictor.Clone() is not concurrency-safe, see
	// https://github.com/PaddlePaddle/Paddle/issues/24887.
	for i := 0; i < size; i++ {
		if err = pool.CreateResource(context.Background()); err != nil {
			panic(err)
		}
	}

	return &PredictorPool{pool: pool}
}

func (p *PredictorPool) Get() (predictor *paddle.Predictor, put func()) {
	resource, err := p.pool.Acquire(context.Background())
	if err != nil {
		panic(err)
	}
	return resource.Value(), func() { resource.Release() }
}
