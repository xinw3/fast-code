package mapred.hashtagsim;

import java.io.IOException;
import mapred.job.Optimizedjob;
import mapred.util.FileUtil;
import mapred.util.SimpleParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

public class Driver {

	public static void main(String args[]) throws Exception {
		SimpleParser parser = new SimpleParser(args);

		String input = parser.get("input");
		String output = parser.get("output");
		String tmpdir = parser.get("tmpdir");

		// output: #a A:2;B:1...
		getHashtagFeatureVector(input, tmpdir + "/feature_vector");
		getHashtagSimilarities(tmpdir + "/feature_vector/", output);
	}

	/**
	 * Same as getJobFeatureVector, but this one actually computes feature
	 * vector for all hashtags.
	 *
	 * @param input
	 * @param output
	 * @throws Exception
	 */
	private static void getHashtagFeatureVector(String input, String output)
			throws Exception {
		Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
				"Get feature vector for all hashtags");
		job.setClasses(HashtagMapper.class, HashtagReducer.class, HashtagCombiner.class);
		job.setMapOutputClasses(Text.class, Text.class);
		job.run();
	}

	/**
	 * When we have feature vector for both #job and all other hashtags, we can
	 * use them to compute inner products. The problem is how to share the
	 * feature vector for #job with all the mappers. Here we're using the
	 * "Configuration" as the sharing mechanism, since the configuration object
	 * is dispatched to all mappers at the beginning and used to setup the
	 * mappers.
	 *
	 * @param jobFeatureVector
	 * @param input
	 * @param output
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 */
	private static void getHashtagSimilarities(String input, String output)
		throws IOException, ClassNotFoundException, InterruptedException {
		// Share the feature vector of #job to all mappers.
		Configuration conf = new Configuration();

		Optimizedjob job = new Optimizedjob(conf, input, output,
				"Get similarities between hashtag pairs");
		job.setClasses(SimilarityMapper.class, SimilarityReducer.class, SimilarityCombiner.class);
		job.setMapOutputClasses(Text.class, IntWritable.class);
		job.run();
	}
}
