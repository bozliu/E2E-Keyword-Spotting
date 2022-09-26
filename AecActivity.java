package com.example.speech_process;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Activity;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.locks.ReentrantLock;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import biz.source_code.dsp.math.Complex;
import biz.source_code.dsp.transform.Dft;

import org.jtransforms.fft.DoubleFFT_1D;

import static java.lang.Math.min;

public class AecActivity extends AppCompatActivity implements View.OnClickListener{
    Button load_wav;
    Button play;
    Button recording;
    Button play_ori;

    boolean shouldContinue = true;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    int recordingOffset = 0;
    private static final int SAMPLE_RATE = 16000;
    // maximum recording time 10s
    private static final int SAMPLE_DURATION_MS = 10000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    short[] recordingBuffer = new short[RECORDING_LENGTH];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_aec);
        Intent intent = getIntent();
        init();
    }

    private void init(){
        System.out.printf("====audio_path %s", getExternalCacheDir().getAbsolutePath());
        // get permissions for record, read & write external storage
        //verifyPermissions(AecActivity.this);
        load_wav = findViewById(R.id.load_aec);
        play = findViewById(R.id.play_aec);
        recording = findViewById(R.id.record_aec);
        //stop = findViewById(R.id.stop);
        play_ori = findViewById(R.id.playori_aec);
        addListener();
    }

    private void addListener(){
        load_wav.setOnClickListener(this);
        play.setOnClickListener(this);
        recording.setOnClickListener(this);
        //stop.setOnClickListener(this);
        play_ori.setOnClickListener(this);
    }

    @Override
    public void onClick(View view){
        switch (view.getId()) {
            case R.id.load_aec:
                Map AEC_audio;
                double[] out_AEC;
                AEC_audio = readAEC_audio();
                play_audio("doubletalk_mic.wav");
                out_AEC = AEC((double[]) AEC_audio.get("mic"), (double[]) AEC_audio.get("lpb"));
                //write_audio(audio);
                break;
            case R.id.record_aec:
                shouldContinue = true;
                play_audio("game.wav");
                record();
                precess_record();
                break;
            case R.id.play_aec:
                play_audio("after.wav");
                break;
            case R.id.playori_aec:
                play_audio("record_audio.wav");
                break;
        }
    }

    private static String[] PERMISSION_ALL = {
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
    };

    public static void verifyPermissions(Activity activity) {
        // check if have all permissions
        boolean permission = (ActivityCompat.checkSelfPermission(activity, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
                || (ActivityCompat.checkSelfPermission(activity, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)
                || (ActivityCompat.checkSelfPermission(activity, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED);
        // if don't have all permissions, requestion permissions
        if (permission) {
            ActivityCompat.requestPermissions(activity, PERMISSION_ALL, 1);
        }
    }

    public Map readAEC_audio() {
        Map<String, double[]> map = new HashMap<String, double[]>();
        try {
            Log.i("=====","read audio ");
            //Log.i("====audio_path %s", getExternalCacheDir().getAbsolutePath());
//            WavFile mic = WavFile.openWavFile(new File("/sdcard"+Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/doubletalk_mic.wav"));
//            WavFile lpb = WavFile.openWavFile(new File("/sdcard"+Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/doubletalk_lpb.wav"));
            WavFile mic = WavFile.openWavFile(new File(getExternalCacheDir().getAbsolutePath() + "/AEC/doubletalk_mic.wav"));
            WavFile lpb = WavFile.openWavFile(new File(getExternalCacheDir().getAbsolutePath() + "/AEC/doubletalk_lpb.wav"));
            //System.out.printf("Environment.getExternalStorageDirectory().getAbsolutePath()", Environment.getExternalStorageDirectory().getAbsolutePath());
            // Display information about the wav file
            System.out.printf("=======wavfile display%d", mic.getNumFrames());
            // wavFile.display();
            int numFrames_mic = (int) mic.getNumFrames();
            int numFrames_lpb = (int) lpb.getNumFrames();
            // Get the number of audio channels in the wav file
            int numChannels_mic = mic.getNumChannels();
            int numChannels_lpb = lpb.getNumChannels();
            // Create a buffer of 100 frames
            double[] buffer_mic = new double[numFrames_mic * numChannels_mic];
            double[] buffer_lpb = new double[numFrames_lpb * numChannels_lpb];
            mic.readFrames(buffer_mic, numFrames_lpb);
            lpb.readFrames(buffer_lpb, numFrames_lpb);
            // Close the wavFile
            mic.close();
            lpb.close();
            map.put("mic", buffer_mic);
            map.put("lpb", buffer_lpb);
            return map;
        } catch (Exception e) {
            System.err.println(e);
        }
        return null;
    }

    public double[] readWav(String name) {
        try {
            //System.out.printf("====audio_path %s", getExternalCacheDir().getAbsolutePath());
            //String path = "/sdcard"+Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/" + name;
            String path = getExternalCacheDir().getAbsolutePath() + "/AEC/"+ name;
            System.out.printf("path", path);
            WavFile wavFile = WavFile.openWavFile(new File(path));
            // Display information about the wav file
            System.out.printf("=======wavfile display%d", wavFile.getNumFrames());
            // wavFile.display();
            int numFrames = (int) wavFile.getNumFrames();
            // Get the number of audio channels in the wav file
            int numChannels = wavFile.getNumChannels();
            // Create a buffer of 100 frames
            double[] buffer = new double[numFrames * numChannels];
            wavFile.readFrames(buffer, numFrames);
            // Close the wavFile
            wavFile.close();
            return buffer;
        } catch (Exception e) {
            System.err.println(e);
        }
        return null;
    }

    private double[] AEC(double[] audio_in, double[] lpb_in){

        //============  Load from recording buffer =============
        System.out.println("======do AEC");
        //Interpreter model1;
        //Interpreter model2;
        int block_len=512;
        int block_shift = 128;
        int len_audio = min(audio_in.length, lpb_in.length);
        double[] audio_init = Arrays.copyOfRange(audio_in, 0, len_audio);
        double[] lpb_init = Arrays.copyOfRange(lpb_in, 0, len_audio);
        int padding = block_len - block_shift;
        double[] audio = new double[padding*2 + len_audio];
        double[] lpb = new double[padding*2 + len_audio];
        for (int i = 0; i < padding; ++i) {
            audio[i] = 0; lpb[i] = 0;}
        for (int i = padding; i <len_audio; ++i) {
            audio[i] = audio_init[i-padding];
            lpb[i] = lpb_init[i-padding];}
        for (int i = len_audio; i <padding*2+len_audio; ++i) {
            audio[i] = 0; lpb[i] = 0;}

        Log.i("======","load tflite");
        // load model
        MappedByteBuffer tfliteModel1 = null;
        MappedByteBuffer tfliteModel2 = null;
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setUseNNAPI(true);
            tfliteModel1 = FileUtil.loadMappedFile(AecActivity.this, "aec_complex.tflite");
            tfliteModel2 = FileUtil.loadMappedFile(AecActivity.this, "aec_2.tflite");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Interpreter.Options options = new Interpreter.Options();
        options.setUseNNAPI(true);
        Interpreter model1 = new Interpreter(tfliteModel1, options);
        Interpreter model2 = new Interpreter(tfliteModel2, options);
        Log.i("=====","init LSTM");
        //create states for LSTM
        float[][][][] states_1 = new float[1][3][1024][2];
        //float[][][][] states_1 = new float[1][2][512][2];
        double[] out_file = new double[len_audio];
        double[] in_buffer = new double[block_len];
        double[] in_buffer_lpb = new double[block_len];
        double[] out_buffer = new double[block_len];
        int num_block = (len_audio - (block_len-block_shift)) / block_shift;
        //int num_block=1;
        Log.i("=====","run model num_block" + num_block);
        Log.i("=====","length audio" + len_audio);
        // Run model
        for (int idx = 0; idx < num_block; ++idx) {
            //0-384
            for (int j = 0; j < in_buffer.length-block_shift; ++j) {
                in_buffer[j] = in_buffer[block_shift + j];
                //Log.i("in_buffer","index= " + idx + "in_audio=" + Arrays.toString(in_buffer));
            }
            //-128-0
            for (int j = in_buffer.length-block_shift,q=0; j < in_buffer.length; ++j,++q) {
                in_buffer[j] = audio_init[idx * block_shift + q];
            }
            for (int j = 0; j < in_buffer_lpb.length-block_shift; ++j) {
                in_buffer_lpb[j] = in_buffer_lpb[block_shift + j];
                //Log.i("in_buffer","index= " + idx + "in_audio=" + Arrays.toString(in_buffer));
            }
            for (int j = in_buffer_lpb.length-block_shift,q=0; j < in_buffer_lpb.length; ++j,++q) {
                in_buffer_lpb[j] = lpb_init[idx * block_shift + q];
            }
            //# calculate fft of input block
            System.out.printf("in_buffer", in_buffer);
            System.out.printf("in_buffer", in_buffer_lpb);

            Scanner scanner = null;
            try {
                //scanner = new Scanner(new File("/sdcard"+ Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/hanning.txt"));
                scanner = new Scanner(new File(getExternalCacheDir().getAbsolutePath() + "/AEC/hanning.txt"));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            double [] hanning = new double [512];
            int index = 0;
            while(scanner.hasNextDouble()){
                hanning[index++] = scanner.nextDouble();
            }

            for (int j = 0; j < in_buffer.length; ++j) {
                in_buffer[j] = in_buffer[j]*hanning[j];
                in_buffer_lpb[j] = in_buffer_lpb[j]*hanning[j];
            }
            Complex[] in_block_fft = Dft.goertzel(in_buffer);
            Complex[] lpb_block_fft = Dft.goertzel(in_buffer_lpb);
            //in_mag = np.abs(in_block_fft) ;
            //Log.i("in_block_fft","index= " + idx + "in_audio=" + Arrays.toString(in_block_fft));
            //Log.i("test","="+in_block_fft.length);
            /*float[][][] in_mag = new float[1][1][257];
            double[] in_phase = new double[257];
            float[][][] lpb_mag = new float[1][1][257];
            double[] lpb_phase = new double[257];*/
            float[][][] in_re = new float[1][1][257];
            float[][][] in_im = new float[1][1][257];
            float[][][] lpb_re = new float[1][1][257];
            float[][][] lpb_im = new float[1][1][257];

            for(int k=0; k<257; ++k) {
                /*
                in_mag[0][0][k] = (float) (in_block_fft[k].abs());
                in_phase[k] = in_block_fft[k].arg();*/
                in_re[0][0][k] = (float) (in_block_fft[k].re());
                in_im[0][0][k] = (float)(in_block_fft[k].im());
            }
            for(int k=0; k<257; ++k) {
                /*lpb_mag[0][0][k] = (float) (lpb_block_fft[k].abs());
                lpb_phase[k] = lpb_block_fft[k].arg();*/
                lpb_re[0][0][k] = (float) (lpb_block_fft[k].re());
                lpb_im[0][0][k] = (float)(lpb_block_fft[k].im());
            }
//            Log.i("in mag","="+ in_re[0][0].length);
//            Log.i("in mag","="+ Arrays.deepToString(in_re));
//            Log.i("lpb_mag","=" + Arrays.deepToString(lpb_re));
//            Log.i("states_1","=" + Arrays.toString(states_1));
            /*Object[] inputs1 = {in_mag, states_1, lpb_mag};

            float[][][] out_mask = new float[1][1][257];*/
            Object[] inputs1 = {in_re,in_im, lpb_re, lpb_im,states_1};
            //Object[] inputs1 = {in_re, states_1, lpb_re};
            float[][][] out_mask_real = new float[1][1][257];
            float[][][] out_mask_imag = new float[1][1][257];
            Map<Integer, Object> outputs1 = new HashMap<>();
            /*outputs1.put(0, out_mask);
            outputs1.put(1, states_1);*/
            outputs1.put(0, out_mask_real);
            outputs1.put(1, out_mask_imag);
            outputs1.put(2, states_1);
            //Log.i("=====","tf1 start");
            if (model1 != null){
                model1.runForMultipleInputsOutputs(inputs1, outputs1);
            }
            //Log.i("=====","tf1 done");
//            Log.i("in_re","index= " + idx + "in_audio=" + Arrays.toString(in_re[0][0]));
//            Log.i("in_im","index= " + idx + "in_audio=" + Arrays.toString(in_im[0][0]));
//            Log.i("lpb_re","index= " + idx + "in_audio=" + Arrays.toString(lpb_re[0][0]));
//            Log.i("lpb_im","index= " + idx + "in_audio=" + Arrays.toString(lpb_im[0][0]));
//            Log.i("out_mask","index= " + idx + "in_audio=" + Arrays.toString(out_mask_real[0][0]));
            //# calculate the ifft
            //estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
            Complex i = Complex.I;
            Complex[] estimated_complex = new Complex [257];
            for (int k = 0; k < 257; ++k) {
                /*
                double tmp;
                Complex tmpp;
                tmp = in_mag[0][0][k]*out_mask[0][0][k];
                tmpp = i.mul(in_phase[k]).exp();
                estimated_complex[k] = tmpp.mul(tmp);*/
                double enh_re,enh_im;
                enh_re = in_re[0][0][k] * out_mask_real[0][0][k] - in_im[0][0][k] * out_mask_imag[0][0][k];
                enh_im=in_re[0][0][k] *out_mask_imag[0][0][k] + in_im[0][0][k] *out_mask_real[0][0][k];
                estimated_complex[k] = i.mul(enh_im).add(enh_re);//new Complex(enh_re, enh_im);
            }

//            Log.i("enh_re",String.valueOf(in_re[0][0][1] * out_mask_real[0][0][1] - in_im[0][0][1] * out_mask_imag[0][0][1]));
//            Log.i("enh_re_review",String.valueOf(estimated_complex[1].re()));

            double[] com = new double[estimated_complex.length*2 - 2];
            for (int k = 0; k < estimated_complex.length-1; ++k) {
                com[2*k] = estimated_complex[k].re();
                com[2*k+1] = estimated_complex[k].im();
            }
            com[1] = estimated_complex[(estimated_complex.length-1)].re();

            DoubleFFT_1D fft = new DoubleFFT_1D(com.length);
            fft.realInverse(com,true);
            double[] estimated_block = com;

            //Log.i("estimated_block_length",String.valueOf(estimated_complex.length));
            //Log.i("estimated_block_length",String.valueOf(estimated_block.length));
            //Log.i("estimated_block","index= " + idx + Arrays.toString(estimated_block));

            float[][][] float_estimated_block = new float[1][1][512];
            for (int k = 0; k < float_estimated_block[0][0].length; ++k) {
                float_estimated_block[0][0][k] = (float) (estimated_block[k]);
            }
            /*float[][][] in_buffer_lpb_2 = new float[1][1][512];
            for (int k = 0; k < in_buffer_lpb_2[0][0].length; ++k) {
                in_buffer_lpb_2[0][0][k] = (float) (in_buffer_lpb[k]);
            }
            Object[] inputs2 = {float_estimated_block, states_2, in_buffer_lpb_2};

            float[][][] out_block = new float[1][1][512];
            Map<Integer, Object> outputs2 = new HashMap<>();
            outputs2.put(0, out_block);
            outputs2.put(1, states_2);
            //Log.i("in mag","="+ float_estimated_block[0][0].length);
            //Log.i("in_buffer_lpb","="+ in_buffer_lpb_2[0][0].length);
            //Log.i("=====","tf2 start");
            if (model2 != null){
                model2.runForMultipleInputsOutputs(inputs2,outputs2);
            }
            //Log.i("=====","tf2 done");
            //# shift values and write to buffer
            //out_buffer[:-block_shift] = out_buffer[block_shift:]
            //Log.i("idx",String.valueOf(idx));*/
            for (int j = 0; j < out_buffer.length-block_shift; ++j) {
                out_buffer[j] = out_buffer[block_shift + j];
            }
            //out_buffer[-block_shift:] = np.zeros((block_shift))
            for (int j = out_buffer.length-block_shift; j < out_buffer.length; ++j) {
                out_buffer[j] = 0;
            }
            //out_buffer  += np.squeeze(out_block)
            for (int j = 0; j < out_buffer.length; ++j) {
                /*out_buffer[j] = out_buffer[j] + out_block[0][0][j];*/
                out_buffer[j] = out_buffer[j] +float_estimated_block[0][0][j];
                //Log.i("out_block","value=" + out_block[0][0][j]);
            }
            //# write block to output file
            //out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
            for (int k = 0; k < block_shift; ++k) {
                out_file[idx*block_shift+k] = out_buffer[k];
            }
            if (idx > 1330){
                Log.i("=====","block" + idx +"done");
            }
        }
        Log.i("=====","predicted_speech");
        Log.i("=====","=" + len_audio);
        Log.i("=====","=" + out_file.length);
        //double[] predicted_speech = new double[len_audio];
//        for (int i = 0; i < len_audio; ++i) {
//            predicted_speech[i] = out_file[i+block_len-block_shift];
//        }
        //double maxx = max(predicted_speech);
        Log.i("=====","call writre audio");
        write_audio(out_file, "after.wav");
        return out_file;
    }

    public void write_audio(double[] audio, String name)
    {
        try
        {
            int sampleRate = 16000;    // Samples per second
            Log.v("write_enter", String.valueOf(sampleRate));
            // Calculate the number of frames required for specified duration
            int numFrames = audio.length;
            // Create a wav file with the name specified as the first argument
            //String path = "/sdcard"+Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/"+ name;
            String path =getExternalCacheDir().getAbsolutePath() + "/AEC/"+ name;
            File outputFile = new File(path);
            Log.v("write_path", String.valueOf(outputFile));
            //ContextWrapper cw = new ContextWrapper(getApplicationContext());
            //File directory = cw.getExternalFilesDir(Environment.DIRECTORY_MUSIC);
            //File outputFile = new File(directory,"after.wav");
            //Log.v("outfile path", String.valueOf(outputFile));
            WavFile wavFile = WavFile.newWavFile(outputFile,
                    1, numFrames, 16, sampleRate);
            Log.v("====", "step2");
            //long frameCounter = 0;
            wavFile.writeFrames(audio, numFrames);
            wavFile.close();

        }
        catch (Exception e)
        {
            System.err.println(e);
        }
    }

    public void play_audio(String name) {
        MediaPlayer mp = new MediaPlayer();
        try {
            //String path = "/sdcard/Android/data/com.example.speech_process/files/Music" + name;
            //String path = "/sdcard"+Environment.getExternalStorageDirectory().getAbsolutePath() + "/AEC/"+ name;
            String path =getExternalCacheDir().getAbsolutePath() +  "/AEC/"+ name;
            Log.v("play_path", path);
            //mp.setDataSource("/sdcard/Android/data/com.example.speech_process/files/Music/after.wav");
            mp.setDataSource(path);
            mp.prepare();
            mp.start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private double[] record() {
        Log.v("====", "recording");
        int bufferSize =
                AudioRecord.getMinBufferSize(
                        16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = 16000 * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        16000,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            System.out.println("Audio Record can't initialize!");
            //return audioBuffer;
        }
        record.startRecording();
        System.out.println("Start recording");
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            //System.out.println("number_read: " + numberRead);
            int maxLength = recordingBuffer.length;
            recordingBufferLock.lock();
            try {
                if (recordingOffset + numberRead < maxLength) {
                    System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, numberRead);
                } else {
                    shouldContinue = false;
                }
                recordingOffset += numberRead;

            } finally {
                recordingBufferLock.unlock();
            }
        }
        record.stop();
        record.release();

        double[] floatInputBuffer = new double[recordingOffset];
        for (int i = 0; i < recordingOffset; ++i) {
            floatInputBuffer[i] = (double) (recordingBuffer[i] / 32768.);
        }
        //double[] after = enhancement(floatInputBuffer);
        write_audio(floatInputBuffer, "record_audio.wav");
        //shouldContinue=true;
        recordingOffset = 0;
        return floatInputBuffer;
    }

    private void precess_record(){
        double[] record_audio = readWav("record_audio.wav");
        double[] lpb = readWav("game.wav");
        double[] after = AEC(record_audio, lpb);
    }


}