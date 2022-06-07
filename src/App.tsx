import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Rank, Tensor } from '@tensorflow/tfjs';
import pica from 'pica';
import Grid from '@mui/material/Grid';
import { Button, Typography } from '@mui/material';

const pic = pica();
const model = await tf.loadLayersModel('model/model.json');

export default function App() {
  const canvasRef = useRef(null as HTMLCanvasElement | null);
  const contextRef = useRef(null as CanvasRenderingContext2D | null);
  const [isDrawing, setIsDrawing] = useState<boolean>();

  const resultCanvas = useRef(null as HTMLCanvasElement | null);

  const [guess, setGuess] = useState<number>();

  useEffect(() => {
    if (!canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 5;
    contextRef.current = ctx;
  }, []);

  useEffect(() => {
    predict();
  }, [isDrawing]);

  const predict = async () => {
    if (!canvasRef.current || !resultCanvas.current || isDrawing !== false)
      return;
    await pic.resize(canvasRef.current, resultCanvas.current);
    const ctx = resultCanvas.current.getContext('2d');
    if (ctx) {
      const imgData = ctx.getImageData(
        0,
        0,
        resultCanvas.current.width,
        resultCanvas.current.height
      );
      const pixels: number[] = [];
      for (let i = 0; i < imgData.data.length; i += 4) {
        const count =
          imgData.data[i] + imgData.data[i + 1] + imgData.data[i + 2];
        if (count > 383) pixels.push(0);
        else pixels.push(1);
      }
      const prediction = tf.tidy(() => {
        const img = tf.tensor([chunkArray(pixels, 28)]);
        const output = model.predict(img) as Tensor<Rank>;
        return Array.from(output.dataSync());
      });
      setGuess(indexOfMax(prediction));
    }
  };

  return (
    <>
      <Grid container spacing={1} justifyContent={'center'}>
        <Grid item xs={12} sx={{ textAlign: 'center' }}>
          <Typography variant={'h5'} children={'Digit Recognition AI'} />
          <Typography
            children={'Author: Norbert Elter'}
            sx={{ cursor: 'pointer' }}
            onClick={() => window.open('https://github.com/itsyoboieltr')}
          />
        </Grid>
        <Grid item xs={0}>
          <canvas
            ref={canvasRef}
            style={{ border: '2px solid black', touchAction: 'none' }}
            width={200}
            height={200}
            onPointerDown={(e) => {
              if (!contextRef.current) return;
              const { offsetX, offsetY } = e.nativeEvent;
              contextRef.current.beginPath();
              contextRef.current.moveTo(offsetX, offsetY);
              setIsDrawing(true);
            }}
            onPointerUp={(e) => {
              if (!contextRef.current) return;
              contextRef.current.closePath();
              setIsDrawing(false);
            }}
            onPointerMove={(e) => {
              if (!isDrawing || !contextRef.current) return;
              const { offsetX, offsetY } = e.nativeEvent;
              contextRef.current.lineTo(offsetX, offsetY);
              contextRef.current.stroke();
            }}
          />
        </Grid>
        <Grid item xs={12} sx={{ textAlign: 'center' }}>
          {guess !== undefined && (
            <>
              <Typography variant={'h5'}>
                My guess is{' '}
                <Typography fontSize={27} color={'primary'} display={'inline'}>
                  {guess}
                </Typography>
              </Typography>
              <Button
                children={'Reset'}
                onClick={() => {
                  if (
                    !contextRef.current ||
                    !canvasRef.current ||
                    !resultCanvas.current
                  )
                    return;
                  contextRef.current.fillRect(
                    0,
                    0,
                    canvasRef.current.width,
                    canvasRef.current.height
                  );
                  const ctx = resultCanvas.current.getContext('2d');
                  if (ctx) {
                    ctx.fillRect(
                      0,
                      0,
                      resultCanvas.current.width,
                      resultCanvas.current.height
                    );
                  }
                  setGuess(undefined);
                }}
              />
            </>
          )}
        </Grid>
      </Grid>
      <canvas
        ref={resultCanvas}
        style={{ marginLeft: 10, border: '2px solid black', display: 'none' }}
        width={28}
        height={28}
      />
    </>
  );
}

const chunkArray = (arr: number[], size: number): (number | number[])[] =>
  arr.length > size
    ? [arr.slice(0, size), ...chunkArray(arr.slice(size), size)]
    : [arr];

function indexOfMax(arr: number[]) {
  if (arr.length === 0) return -1;
  let max = arr[0];
  let maxIndex = 0;
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) {
      maxIndex = i;
      max = arr[i];
    }
  }
  return maxIndex;
}
